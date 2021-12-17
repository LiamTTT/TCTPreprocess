# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: sampling_PMScope.py
@Date: 2021/9/22 
@Time: 下午12:12
@Desc: Create jpeg image according to Annotation xml files of PMScope.
'''
import os
import json
import argparse
from time import time
from threading import Thread
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, as_completed)

import cv2
import xmltodict
import numpy as np
from loguru import logger
from glob2 import glob

from core.sampling import sampling
from core.common import check_dir, verify_jpeg

SRC_TYPE = 'PortableMicroScope'


def _save_image(image_path, image, max_try=3):
    for _ in range(max_try):
        if verify_jpeg(image_path):
            return True
        cv2.imwrite(image_path, image)
    if verify_jpeg(image_path):
        return True
    else:
        # delete corrupt image and return false
        os.remove(image_path)
        return False


# TODO 210928 siboliu: move those function to core.sampling module, using class to save global vars.
def create_sample_for_annotation_xml(xml_file):
    annotation = xmltodict.parse(open(xml_file, 'rb'))['annotation']
    anno_src = annotation['source']
    roi_path = BMP_INFO_LUT[anno_src['image_roi']]['path']
    roi_bnd = anno_src['wsi_bndbox']
    roi_bnd = [roi_bnd['xmin'], roi_bnd['ymin'], roi_bnd['xmax'], roi_bnd['ymax']]  # make sure the order is correct.
    roi_bnd = np.array(list(map(float, roi_bnd)), dtype=int)
    roi_handle = cv2.imread(roi_path)
    try:
        return sampling(roi_bnd, roi_handle, SRC_TYPE)
    except:
        logger.exception(f'create sample for {xml_file} failed!')
        return None


def create_and_save_sample_for_annotation_xml(xml_file, jpeg_path, lock=None):
    if os.path.exists(jpeg_path):
        jpeg_status = verify_jpeg(jpeg_path)
        if jpeg_status:
            return True
        logger.info(f'{jpeg_path} has been corrupted, remake.')
    sample = create_sample_for_annotation_xml(xml_file)
    if sample is None:
        logger.error(f'create {jpeg_path} for {xml_file} failed!')
        return False
    if not _save_image(jpeg_path, sample):
        logger.error(f'save {jpeg_path} for {xml_file} failed!')
        return False
    return True


def create_and_save_samples_for_dataset(dataset_root, dataset_name):
    since = time()
    dataset_dir = os.path.join(dataset_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        logger.error(f'{dataset_dir} is not a valid directory')
        return False
    logger.info(f'process {dataset_name}')
    xml_files = glob(os.path.join(dataset_dir, 'Annotations', '*.xml'))
    logger.info(f'annotations: {len(xml_files)}')
    dataset_jpegs = os.path.join(dataset_dir, 'JPEGImages')
    if check_dir(dataset_jpegs, create=True, logger=logger):
        logger.info(f'JPEGs save in {dataset_jpegs}')
    else:
        logger.error(f'create JPEG directory failed: {dataset_jpegs}')
        return False
    jpeg_files = [os.path.join(dataset_jpegs, os.path.basename(f).replace('xml', 'jpg')) for f in xml_files]
    logger.info(f'jpeg samples: {len(jpeg_files)}')
    # parallel
    nb_success = 0
    # lock = Lock()
    with ThreadPoolExecutor(5, f'{dataset_name}-ThreadPool') as executor:
        future_to_xml = {
            executor.submit(
                create_and_save_sample_for_annotation_xml,
                xml_files[i], jpeg_files[i]
            ): xml_files[i]
            for i in range(len(xml_files))
        }
        for future in future_to_xml:
            xml_file = future_to_xml[future]
            try:
                res = future.result()
            except Exception as exc:
                logger.exception(f'{exc} when process {xml_file}')
            else:
                nb_success += int(res)
    logger.info(f'success jpeg images: {nb_success}')
    logger.info(f'{dataset_name} time consuming: {(time()-since)/60} mins')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create jpeg image according to Annotation xml files.')
    parser.add_argument('-r', '--dataset_root', help='dir to all PMScope dataset root', type=str)
    parser.add_argument('-l', '--dataset_list', help='dataset name list, also the folder name. Empty for all',
                        type=str, nargs='*', default=[])
    parser.add_argument('-j', '--json_lut_file', help='path to PMScope ROI info LUT file', type=str)
    args = parser.parse_args()

    # parse lut
    lut_file = os.path.abspath(args.json_lut_file)
    global BMP_INFO_LUT
    BMP_INFO_LUT = json.load(open(lut_file, 'r'))
    # end parse lut

    # parse args
    dataset_root = os.path.abspath(args.dataset_root)
    dataset_list = args.dataset_list

    if not os.path.isdir(dataset_root):
        raise ValueError(f'Dataset root: {dataset_root} is not a directory')
    logger.info(f'Dataset root: {dataset_root}')

    if len(dataset_list) == 0:
        dataset_list = [fd for fd in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, fd))]
        logger.info(f'Create all dataset in {dataset_root}')
    logger.info(f'Datasets: {dataset_list}')
    # end parse args

    # process
    with ProcessPoolExecutor(8) as ppool:
        future_to_dataset = {
            ppool.submit(
                create_and_save_samples_for_dataset,
                dataset_root, dataset_name
            ): dataset_name
            for dataset_name in dataset_list
        }
        for future in future_to_dataset:
            dataset_name = future_to_dataset[future]
            try:
                res = future.result()
            except Exception as exc:
                logger.exception(f'Process Dataset-{dataset_name} failed!')
            else:
                if res:
                    logger.info(f'Finish Dataset-{dataset_name}!')
                else:
                    logger.error(f'Process Dataset-{dataset_name} failed!')
    logger.info('Finish ALL!')
    # end process