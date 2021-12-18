# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Project: TCTPreprocess
@File: cvt2voc_PMScope.py
@Date: 2021/9/15 
@Time: 上午10:16
@Desc: convert PMScope csv annotation file to voc format with specified hyper params.
'''
import os
import json
import argparse
import random
from time import time

import numpy as np
from tqdm import tqdm
from loguru import logger

from core.common import check_dir, save_xml_str
from core.annotations import (
    find_neighbors_for_given_center, get_bnd_in_tile,
    create_voc_annotation, create_voc_object,
    BndBox
)

random.seed(42)


def process_csv_file(csv_file):
    annotations = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]
        logger.info(f'Num of annotations: {len(lines)}')
        nb_success = 0
        for line in lines:
            bat, img_roi, cx, cy, w, h, anno_n, sld_n, scx, scy, roi_p = line.strip().split(',')
            # rename batch name
            try:
                if bat.upper() not in BMP_INFO_LUT[img_roi]['batch']:
                    logger.error(
                        'slide batch({}) does not match to json file({}).'.format(bat, BMP_INFO_LUT[img_roi]['batch']))
                    continue
                bat = BMP_INFO_LUT[img_roi]['batch']
            except KeyError:
                logger.exception(f'No {roi_p} info in BMP_INFO_LUT')
                continue
            # init data struct, if necessary
            if bat not in annotations.keys():
                annotations[bat] = {}
            if sld_n not in annotations[bat].keys():
                annotations[bat][sld_n] = {}
            if img_roi not in annotations[bat][sld_n].keys():
                annotations[bat][sld_n][img_roi] = {
                    'ROI_path': roi_p,
                    'objects': []
                }
            # record
            obj = {
                'name': anno_n,
                'center': [int(float(cx)), int(float(cy))],
                'bnd_size': [int(float(w)), int(float(h))],
                'wsi_center': [int(float(scx)), int(float(scy))]
            }
            annotations[bat][sld_n][img_roi]['objects'].append(obj)
            nb_success += 1
    logger.info(f'{nb_success} success')
    return annotations


def get_obj_in_tile(name, center_delta, bnd_size, tile_size):
    (xmin, ymin, xmax, ymax), seg = get_bnd_in_tile(center_delta, bnd_size, tile_size)
    if xmin == xmax or ymin == ymax:
        # FIXME 211217 siboliu: The main reason for zero obj is not clear, this condition is temporal solution.
        return None, None
    return create_voc_object(name, xmin, ymin, xmax, ymax), seg


def create_annotation_of_roi(image_roi, center_anno=True, nb_sample=1):
    annotations = []
    # get roi meta info
    roi_info = BMP_INFO_LUT[image_roi]
    # source info
    image_device = 'PortableMicroScope'
    wsi_batch = roi_info['batch']
    wsi_name = roi_info['slide']  # e.g. sfy1104135
    source_kwarg = {'source': {'image_roi': image_roi}}
    folder = wsi_batch
    # filename

    # process object
    annotation_objects = ANNOTATIONS[wsi_batch][wsi_name][image_roi]['objects']
    tile_size = np.array(TILE_SIZE)
    # wsi_bnds_center = np.array([obj['wsi_center'] for obj in annotation_objects])  # bnd center on wsi where the roi comes from
    bnds_center = np.array([obj['center'] for obj in annotation_objects])
    bnds_wh = np.array([obj['bnd_size'] for obj in annotation_objects])
    bnds_cls = [obj['name'] for obj in annotation_objects]
    # create voc annotations
    for idx in range(len(annotation_objects)):
        # get main object
        main_shifts = []
        for shift_id in range(nb_sample):
            shift_x = (random.random() - 0.5) * 0.90 * tile_size[0]
            shift_y = (random.random() - 0.5) * 0.90 * tile_size[1]
            main_shifts.append(np.array([shift_x, shift_y], dtype=int))
        if center_anno:
            main_shifts[0] = np.zeros((2,))
        for shift_id, main_shift in enumerate(main_shifts):
            objects = []
            obj_in_tile, segmented = get_obj_in_tile(bnds_cls[idx], main_shift, bnds_wh[idx], tile_size)
            if obj_in_tile is None:
                logger.warning(f"annotations is invalid: {image_roi}, {idx} annotation, shift id: {shift_id}, bnd: {bnds_wh[idx]}")
                break
            objects.append(obj_in_tile)
            neighbor_idx, neighbor_delta = find_neighbors_for_given_center(idx, main_shift, tile_size, bnds_center)
            for ni, nd in zip(neighbor_idx, neighbor_delta):
                obj_in_tile, seg = get_obj_in_tile(bnds_cls[ni], main_shift + nd, bnds_wh[ni], tile_size)
                if obj_in_tile is None:
                    continue
                segmented |= seg
                objects.append(obj_in_tile)
            # define other attributes in annotation file.
            filename = '{:s}_{:s}_{:0>3d}_{:d}_{:d}_{:d}_{:s}.jpg'.format(wsi_name, image_roi.split('.bmp')[0], idx, shift_id, *bnds_center[idx], bnds_cls[idx])
            padding = [0, 0]  # pad to right and bottom half of tile
            if tile_size[0] % 2:
                padding[0] += 1
            if tile_size[1] % 2:
                padding[1] += 1
            wsi_bndbox = BndBox(
                *((bnds_center[idx] - main_shift - tile_size // 2).tolist() + (bnds_center[idx] - main_shift + (tile_size // 2) + padding).tolist()))
            voc_anno = create_voc_annotation(
                folder, filename,
                image_device, wsi_batch, wsi_name, wsi_bndbox,
                *tile_size,
                objects,
                segmented,
                **source_kwarg
            )
            annotations.append(voc_anno)
    return annotations


def create_annotation_of_batch(batch_name, center_anno=True, nb_sample=1):
    wsis_anno = ANNOTATIONS[batch_name]
    logger.info(f'Process {batch_name}')
    logger.info(f'WSI nums: {len(wsis_anno)}')
    logger.info(f'ROIS nums: {sum([len(r) for _, r in wsis_anno.items()])}')
    nb_success = 0
    batch_annotations = []
    since = time()
    for sld_n, rois_anno in tqdm(wsis_anno.items(), desc=f'Process {batch_name}'):
        for roi_n in rois_anno.keys():
            try:
                roi_annotations = create_annotation_of_roi(roi_n, center_anno, nb_sample)
            except:
                logger.exception('process {}-{} failed! ROI path: {}'.format(sld_n, roi_n, BMP_INFO_LUT[roi_n]['path']))
                continue
            batch_annotations += roi_annotations
            nb_success += 1
    logger.info(f'{nb_success} ROIs success!')
    logger.info(f'Total annotations: {len(batch_annotations)}')
    total_time = time() - since
    logger.info('Time: {:.2f} s, {:.3f} s/ROI'.format(total_time, (total_time / nb_success) if nb_success else float('inf')))
    return batch_annotations


def create_dataset_for_batch(dataset_dir, dataset_name, center_anno=True, nb_sample=1):
    logger.info(f'Create dataset Annotations file: {dataset_dir}')
    save_annos_dir = os.path.join(dataset_dir, 'Annotations')
    check_dir(save_annos_dir, True, logger)
    batch_annotations = create_annotation_of_batch(dataset_name, center_anno, nb_sample)
    nb_success = 0
    since = time()
    for anno in tqdm(batch_annotations, desc=f'Save annotations in {save_annos_dir}'):
        save_path = os.path.join(save_annos_dir, anno.filename.replace('jpg', 'xml'))
        if save_xml_str(anno.to_xml(), save_path, 'w+', logger):
            nb_success += 1
    total_time = time() - since
    logger.info(f'create {nb_success} annotations file in {save_annos_dir}')
    logger.info('Time: {:.2f} s {:.3f} s/file'.format(total_time, (total_time / nb_success) if nb_success else float('inf')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FCDD xml to VOC annotation files.')
    parser.add_argument('--csv_anno_file', help='path to merged PMScope annotations csv file', type=str)
    parser.add_argument('--json_lut_file', help='path to PMScope ROI info LUT file', type=str)
    parser.add_argument('--dataset_root', help='dir to all PMScope dataset root', type=str)
    parser.add_argument('--dataset_list', help='dataset name list, also the folder name. Empty for all',
                        type=str, nargs='*', default=[])
    parser.add_argument('--tile_size', help='tile size in 0.67um/pixel',
                        type=int, nargs='+', default=[800])
    parser.add_argument('--center_anno', default=False, action='store_true', help='always create center annotation')
    parser.add_argument('--nb_sample', default=1, type=int, help='number of sample to create.')
    args = parser.parse_args()

    # Parse args
    dataset_root = args.dataset_root
    # process global var
    lut_file = args.json_lut_file
    global BMP_INFO_LUT
    BMP_INFO_LUT = json.load(open(lut_file, 'r'))
    # get formatted annotations dict
    csv_file = args.csv_anno_file
    global ANNOTATIONS
    ANNOTATIONS = process_csv_file(csv_file)
    # make tile size global
    global TILE_SIZE
    tile_size = args.tile_size
    if len(tile_size) == 1:
        tile_size = tile_size * 2
    elif len(tile_size) == 2:
        pass
    else:
        raise ValueError(f'Tile size {tile_size} should be [width, height] or [length] for square')
    TILE_SIZE = tile_size
    center_anno = args.center_anno
    nb_sample = args.nb_sample
    if nb_sample < 1 or nb_sample > 9:
        nb_sample = np.clip(nb_sample, 1, 9)
        logger.warning(f"clip nb_sample to {nb_sample}")
    logger.info(f'create {nb_sample} samples for each annotation. make annotaion at center: {center_anno}')
    # get dataset list
    dataset_list = args.dataset_list
    if len(dataset_list) == 0:
        dataset_list = list(ANNOTATIONS.keys())
        logger.info(f'Create all dataset in {csv_file}')
    logger.info(f'Datasets: {dataset_list}')
    logger.info(f'Tile Size: {tile_size}')
    # end parse args

    # process
    for dataset_name in dataset_list:
        dataset_dir = os.path.join(dataset_root, dataset_name)
        check_dir(dataset_dir, True, logger)
        create_dataset_for_batch(dataset_dir, dataset_name, center_anno, nb_sample)
    # end process
