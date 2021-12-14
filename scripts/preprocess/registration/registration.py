# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: registration.py
@Date: 2021/12/9 
@Time: 上午10:05
@Desc: Registration for PMScope ROI to WSI
'''
import os
import argparse
import json
import time
import math
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import numpy as np
from glob2 import glob
from loguru import logger

from core.slide_reader import SlideReader, mpp_transformer
from core.common import check_dir, check_file, verify_png

# for xfeature
PMSCOPE_MPP = 0.67
DOWN_LEVEL = 3


def _read_and_xfeature_sift(img_handle, sift, lock, base_mpp, base_ratio, src_mpp=0.67, down_level=3):
    if isinstance(img_handle, SlideReader):
        dst_size = (img_handle.attrs["width"], img_handle.attrs["height"])
        lock.acquire()
        image = img_handle.get_tile((0, 0), dst_size, down_level)  # bgr
        lock.release()
        if "sdpc" in img_handle.suffix:
            image = image[..., ::-1]  # to bgr
    elif isinstance(img_handle, str) and check_file(img_handle):
        image = cv2.imread(img_handle)  # bgr
        scale_ratio = src_mpp/(base_mpp*base_ratio**down_level)
        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"type of image handle {img_handle} is not supported.")
    # TODO Is lock necessary ?
    # lock.acquire()
    kp, des = sift.detectAndCompute(image, None)
    # lock.release()
    img_h, img_w = image.shape[:2]
    del image
    if isinstance(img_handle, SlideReader):
        return kp, des
    else:
        return kp, des, img_w, img_h  # w, h


def _create_matcher():
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _get_homography(target_feature, temp_feature, matcher, abort_thre=0.6, min_match=20):
    target_kp, target_des = target_feature
    temp_kp, temp_des = temp_feature
    matches = matcher.knnMatch(temp_des, target_des, k=2)
    # remove wrong
    good = []
    for m, n in matches:
        if m.distance <= abort_thre * n.distance:
            good.append(m)
    # calculate homography
    if len(good) > min_match:
        temp_pts = np.float32([temp_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target_pts = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(temp_pts, target_pts, cv2.RANSAC, 5.0)
        return M
    else:
        # process failed roi
        return False


def _match_and_save(matcher, target_feature, temp_info, lock, wsi_reader, roi_path, save_path):
    temp_kp, temp_des, temp_w, temp_h = temp_info
    M = _get_homography(target_feature, (temp_kp, temp_des), matcher, ABORT_THRE, MIN_MATCH_COUNT)
    if M is False:
        # todo process failed roi
        return M
    # compute temp_contour
    temp_contour = np.float32([
        [0, 0],
        [0, temp_h - 1],
        [temp_w - 1, temp_h - 1],
        [temp_w - 1, 0]
    ]).reshape(-1, 1, 2)
    target_contour = cv2.perspectiveTransform(temp_contour, M)

    # save WSI cropped image size similar with PMScope ROI
    wsi_mpp = wsi_reader.attrs["mpp"]
    wsi_level_ratio = wsi_reader.attrs["level_ratio"]
    target_contour = mpp_transformer(
        target_contour.ravel(),
        wsi_mpp * wsi_level_ratio ** DOWN_LEVEL,
        wsi_mpp
    )
    target_contour = np.reshape(target_contour, (-1, 2))
    target_rect = cv2.boundingRect(target_contour)
    # crop in the closest level
    crop_level = int(math.log(PMSCOPE_MPP/wsi_mpp, wsi_level_ratio))

    target_rect = np.array(target_rect)
    target_rect = np.clip(target_rect, 0, None)
    lock.acquire()
    target_image = wsi_reader.get_tile(target_rect[:2], target_rect[2:], crop_level)
    lock.release()
    if "sdpc" in wsi_reader.suffix:
        target_image = target_image[..., ::-1]  # to bgr
    # Fine Tune
    temp_image = cv2.imread(roi_path)
    temp_h, temp_w = temp_image.shape[:2]
    sift = cv2.SIFT_create()
    target_feature = sift.detectAndCompute(target_image, None)
    temp_feature = sift.detectAndCompute(temp_image, None)
    M_reverse = _get_homography(temp_feature, target_feature, matcher, ABORT_THRE-0.05, MIN_MATCH_COUNT-10)  # a more restrict rule
    if M_reverse is False:
        logger.warning(f"fine tune failed!")
        return M_reverse
    target_image = cv2.warpPerspective(target_image, M_reverse, (temp_w, temp_h))
    cv2.imwrite(save_path, target_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    lock.acquire()
    with open(os.path.dirname(save_path) + ".txt", "a") as f:
        f.write(f"{save_path};{target_rect.tolist()};{M_reverse.tolist()}\n")
    lock.release()
    del temp_image
    del target_image
    return True


def ensure_related_file_existed(
        sld_n, wsi_bat_suffix,
        roi_ind_bat_dir, wsi_bat_dirs,
        wsi_bat_lut,
        logger
):
    roi_ind_sld_file = os.path.join(roi_ind_bat_dir, f"{sld_n}.txt")
    check_file(roi_ind_sld_file, True, logger)
    # check all the wsi dirs
    for wsi_bat_dir in wsi_bat_dirs:
        wsi_path = os.path.join(wsi_bat_dir, f"{sld_n}.{wsi_bat_suffix}")
        if check_file(wsi_path, False, mute=True):
            # break, find the wsi when the first time
            break
    check_file(wsi_path, True, logger)  # this operation ensure the existing of wsi file
    # get wsi attrs
    wsi_attrs = wsi_bat_lut[f"{sld_n}.{wsi_bat_suffix}"]
    return roi_ind_sld_file, wsi_path, wsi_attrs


def xfeature_sift(sld_n, roi_paths, wsi_reader, lock):
    # prepare for sift
    sift = cv2.SIFT_create()
    # multithread processing
    xfeature_results = {}
    nb_success_img = 0
    since = time.time()
    with ThreadPoolExecutor(THREAD_NUMBER, f'{sld_n}-SIFT-ThreadPool') as executor:
        future_rd_sift = {
            executor.submit(
                _read_and_xfeature_sift,
                wsi_reader, sift, lock, wsi_reader.attrs["mpp"], wsi_reader.attrs["level_ratio"], PMSCOPE_MPP, DOWN_LEVEL
            ): wsi_reader.path
        }  # submit a WSI
        future_rd_sift.update({
            executor.submit(
                _read_and_xfeature_sift,
                roi_p, sift, lock, wsi_reader.attrs["mpp"], wsi_reader.attrs["level_ratio"], PMSCOPE_MPP, DOWN_LEVEL
            ): roi_p
            for roi_p in roi_paths
        })  # submit ROIs
        for future in future_rd_sift:
            img_path = future_rd_sift[future]
            try:
                xfeature_results[img_path] = future.result()
            except Exception as exc:
                logger.exception(f"{exc} when process {img_path}")
            else:
                nb_success_img += 1
    logger.info(f"read and xfeature consume {time.time()-since} s")
    logger.info(f"xfeature success {nb_success_img}/{len(roi_paths) + 1}")
    return xfeature_results


def match_and_save(wsi_reader, target_feature, templates_dict, save_dir, lock, save_homo=False, save_features=False):
    matcher = _create_matcher()
    nb_success = 0
    since = time.time()
    with ThreadPoolExecutor(THREAD_NUMBER, f'{sld_n}-SIFT-ThreadPool') as executor:
        future_mat_save = {
            executor.submit(
                _match_and_save,
                matcher, target_feature, temp_info, lock,
                wsi_reader, roi_path, os.path.join(save_dir, os.path.splitext(os.path.basename(roi_path))[0] + ".png")
            ): roi_path
            for roi_path, temp_info in templates_dict.items()
        }
        for future in future_mat_save:
            roi_path = future_mat_save[future]
            try:
                res = future.result()
            except Exception as exc:
                logger.exception(f'{exc} when process {roi_path}')
            else:
                if not res:
                    logger.warning(f"match {roi_path} failed.")
                nb_success += int(res)
    logger.info(f'match and crop consuming: {(time.time()-since)} s')
    logger.info(f'success images: {nb_success}/{len(templates_dict)}')


def registration(sld_n, wsi_path, wsi_attrs, roi_ind_sld_file, save_dir):
    # get wsi reader
    wsi_reader = SlideReader()
    wsi_reader.open(wsi_path)
    wsi_reader.attrs = wsi_attrs  # linux can not read attrs in sdpc format.
    lock = Lock()
    # get roi path
    with open(roi_ind_sld_file, 'r') as f:
        roi_paths = [l.strip() for l in f.readlines()]
    if RESUME:
        roi_paths = [p for p in roi_paths if not verify_png(os.path.join(save_dir, os.path.basename(p).split('.')[0] + ".png"))]
        if len(roi_paths):
            logger.info(f"resume registration: {batch} {sld_n} left ROI: {len(roi_paths)}")
        else:
            logger.info(f"resume registration: no ROI left in {batch} {sld_n}, skip.")
            return

    # xfeature
    xfeature_results = xfeature_sift(sld_n, roi_paths, wsi_reader, lock)
    # prepare match and save
    target_path = [k for k, v in xfeature_results.items() if len(v) == 2][0]
    target_feature = xfeature_results.pop(target_path)
    templates_dict = xfeature_results.copy()
    # match and save
    match_and_save(wsi_reader, target_feature, templates_dict, save_dir, lock)
    wsi_reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Registration for PMScope ROI and bounding box to WSI.")
    parser.add_argument("-b", "--batches", type=str, default=[], nargs=argparse.ZERO_OR_MORE,
                        help="batch of PMScope data to register. None for all")
    parser.add_argument("-r", "--index_root", type=str, help="root to PMScope ROI index files.")
    parser.add_argument("-s", "--save_root", type=str, help="save root")
    parser.add_argument("-w", "--wsi_lut", type=str, help="WSI attr LUT file.")
    parser.add_argument("-x", "--batch2x", type=str, help="Batch to x file for WSI path")
    parser.add_argument("-t", "--thread_num", type=int, default=2, help="multithread")
    parser.add_argument("-p", "--process_num", type=int, default=2, help="multiprocess")
    parser.add_argument("--abort_threshold", type=float, default=0.5, help="max distance threshold ratio")
    parser.add_argument("--min_match", type=int, default=20, help="min match count threshold.")
    parser.add_argument("--resume", default=False, action="store_true")
    args = parser.parse_args()

    # === Parse Args ===
    # read static vars
    wsi_lut = os.path.abspath(args.wsi_lut)
    with open(wsi_lut, 'r') as f:
        WSI_ATTR_LUT = json.load(f)
    logger.info(f"read WSI attributes LUT: {args.wsi_lut}")
    batch2x = os.path.abspath(args.batch2x)
    with open(batch2x, 'r') as f:
        BATCH2X = json.load(f)
    logger.info(f"read batch to X file for WSI path: {args.batch2x}")

    index_root = os.path.abspath(args.index_root)
    logger.info(f"Use PMScope ROI index files in: {index_root}")
    save_root = os.path.abspath(args.save_root)
    check_dir(save_root, create=True)
    logger.info(f"Cropped WSI ROI will save in: {save_root}")

    batches = [d for d in os.listdir(index_root) if os.path.isdir(os.path.join(index_root, d))]
    if len(args.batches):
        batches = args.batches
    logger.info(f"process batches: {batches}")

    global THREAD_NUMBER
    THREAD_NUMBER = args.thread_num
    global PROCESS_NUMBER
    PROCESS_NUMBER = args.process_num
    # for match
    global ABORT_THRE
    global MIN_MATCH_COUNT
    ABORT_THRE = args.abort_threshold
    MIN_MATCH_COUNT = args.min_match
    # resume
    global RESUME
    RESUME = args.resume
    # === End Parse Args ===

    for batch in batches:
        since = time.time()
        logger.info(f"====== Start process {batch} ======")
        # Batch Process
        roi_ind_bat_dir = os.path.join(index_root, batch)
        sld_names = [os.path.basename(f).split('.')[0] for f in sorted(glob(os.path.join(roi_ind_bat_dir, "*.txt")))]
        # sld_names = ["sfy1630", "sfy3815", "sfy4096"]  # for test
        wsi_bat_rt = BATCH2X["BAT_TO_ROOT"][batch]
        wsi_bat_rel_dir = BATCH2X["BAT_TO_RELATED_DIR"][batch]
        if isinstance(wsi_bat_rel_dir, str):
            wsi_bat_dirs = [os.path.join(wsi_bat_rt, wsi_bat_rel_dir)]
        elif isinstance(wsi_bat_rel_dir, list):
            wsi_bat_dirs = [os.path.join(wsi_bat_rt, d) for d in wsi_bat_rel_dir]
        else:
            raise ValueError(f"Format of WSI related dir in batch {batch} is wrong.")
        wsi_bat_suffix = BATCH2X["BAT_TO_SUFFIX"][batch]
        wsi_bat_lut = WSI_ATTR_LUT[batch]
        bat_save_root = os.path.join(save_root, batch)
        logger.info(f"Total number: {len(sld_names)}")
        logger.info(f"PMScope ROI index files root: {roi_ind_bat_dir}")
        logger.info(f"WSI file({wsi_bat_suffix}) root: {wsi_bat_dirs}")
        logger.info(f"save results in {bat_save_root}")
        # multiprocess for each slide-level registration
        # multithread for each roi-level registration
        nb_finish = 0
        if PROCESS_NUMBER > 1:
            logger.info(f"Multiprocess: {PROCESS_NUMBER}")
            with ProcessPoolExecutor(PROCESS_NUMBER) as ppool:
                future_to_batch = {}
                for sld_idx, sld_n in enumerate(sld_names):
                    logger.info(f"start {batch} [{sld_idx+1}/{len(sld_names)}] {sld_n}")
                    # ensure the file and attr is existed.
                    try:
                        roi_ind_sld_file, wsi_path, wsi_attrs = ensure_related_file_existed(
                                sld_n, wsi_bat_suffix,
                                roi_ind_bat_dir, wsi_bat_dirs,
                                wsi_bat_lut,
                                logger
                        )
                    except FileNotFoundError or KeyError as e:
                        logger.error(f"Batch {batch} Slide {sld_n}"
                                     f"\nWSI Path {wsi_path}"
                                     f"\nIndex file {roi_ind_sld_file}"
                                     f"\n{e}")
                    save_dir = os.path.join(bat_save_root, sld_n)
                    check_dir(save_dir, True)
                    # Process a slide
                    future_to_batch[
                        ppool.submit(
                            registration,
                            sld_n, wsi_path, wsi_attrs, roi_ind_sld_file, save_dir
                        )
                    ] = sld_n
                for future_idx, future in enumerate(future_to_batch):
                    sld_name = future_to_batch[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.exception(f'process {batch} {future_idx+1}th failed!')
                    else:
                        logger.info(f"finish {batch} [{future_idx + 1}/{len(sld_names)}]")
                        nb_finish += 1
        else:
            for sld_idx, sld_n in enumerate(sld_names):
                logger.info(f"start {batch} [{sld_idx + 1}/{len(sld_names)}] {sld_n}")
                # ensure the file and attr is existed.
                try:
                    roi_ind_sld_file, wsi_path, wsi_attrs = ensure_related_file_existed(
                        sld_n, wsi_bat_suffix,
                        roi_ind_bat_dir, wsi_bat_dirs,
                        wsi_bat_lut,
                        logger
                    )
                except FileNotFoundError or KeyError as e:
                    logger.error(f"Batch {batch} Slide {sld_n}"
                                 f"\nWSI Path {wsi_path}"
                                 f"\nIndex file {roi_ind_sld_file}"
                                 f"\n{e}")
                save_dir = os.path.join(bat_save_root, sld_n)
                check_dir(save_dir, True)
                try:
                    registration(sld_n, wsi_path, wsi_attrs, roi_ind_sld_file, save_dir)
                except Exception as exc:
                    logger.exception(f'process {batch} {sld_idx + 1}th failed!')
                else:
                    logger.info(f"finish {batch} [{sld_idx + 1}/{len(sld_names)}]")
                    nb_finish += 1
        if nb_finish:
            logger.info(f"Finish {batch}, {nb_finish} slides")
            time_consume = time.time() - since
            logger.info(f"total time: {time_consume / 3600} hr, avg time: {time_consume / nb_finish / 60} mins/sld")
        else:
            logger.warning(f"No success slides in {batch}")

