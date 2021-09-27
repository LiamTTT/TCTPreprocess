# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: sampling_FCDD.py
@Date: 2021/9/17 
@Time: 上午9:42
@Desc: Create jpeg image according to Annotation xml file.
'''
import os
import json
import argparse
from time import time
from threading import Thread, Lock
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, as_completed)

import cv2
import xmltodict
import numpy as np
from glob2 import glob
from loguru import logger

from core.slide_reader import SlideReader
from core.sampling import sampling
from core.common import check_dir

wsi_handle = SlideReader()
SRC_TYPE = 'MicroScope'


def create_sample_for_annotation_xml(xml_file):
    annotation = xmltodict.parse(open(xml_file, 'rb'))['annotation']
    anno_src = annotation['source']
    wsi_dir = BAT_TO_DIR[anno_src['wsi_batch']]
    wsi_name = anno_src['wsi_name']
    wsi_bnd = anno_src['wsi_bndbox']
    wsi_bnd = [wsi_bnd['xmin'], wsi_bnd['ymin'], wsi_bnd['xmax'], wsi_bnd['ymax']]  # make sure the order is correct.
    wsi_bnd = np.array(list(map(float, wsi_bnd)), dtype=int)
    wsi_path = os.path.join(SLIDE_ROOT, wsi_dir, wsi_name)
    wsi_handle.open(wsi_path)
    try:
        return sampling(wsi_bnd, wsi_handle, SRC_TYPE)
    except:
        logger.exception(f'create sample for {xml_file} failed!')
        return None


def create_and_save_sample_for_annotation_xml(xml_file, jpeg_path, lock):
    if os.path.exists(jpeg_path):
        return True
    with lock:  # make sure the tile is correct
        sample = create_sample_for_annotation_xml(xml_file)
    if sample is None:
        logger.error(f'create {jpeg_path} for {xml_file} failed!')
        return False
    if not cv2.imwrite(jpeg_path, sample):
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
    lock = Lock()
    with ThreadPoolExecutor(5, f'{dataset_name}-ThreadPool') as executor:
        future_to_xml = {
            executor.submit(
                create_and_save_sample_for_annotation_xml,
                xml_files[i], jpeg_files[i], lock
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
    parser.add_argument('--dataset_root', help='dir to all FCDD dataset root', type=str)
    parser.add_argument('--dataset_list', help='dataset name list, also the folder name. Empty for all',
                        type=str, nargs='*', default=[])
    parser.add_argument('--slide_root', help='dir to WSI files', type=str)
    parser.add_argument('--batch_to_x', help='path to batch to x file', type=str)
    parser.add_argument('--attrs_lut', help='path to WSI attributes', type=str)
    parser.add_argument('--log', help='path to log file', type=str, default=f'log/{os.path.basename(__file__)}.log')
    args = parser.parse_args()

    global logger
    logger.add(args.log, backtrace=True, diagnose=True)
    # parse lut
    global BAT_TO_DIR
    global BAT_TO_SUFFIX
    global ATTRS_LUT
    BAT_TO_X = json.load(open(args.batch_to_x, 'r'))
    BAT_TO_DIR = BAT_TO_X['BAT_TO_DIR']
    BAT_TO_SUFFIX = BAT_TO_X['BAT_TO_SUFFIX']
    ATTRS_LUT = json.load(open(args.attrs_lut, 'r'))
    # end parse lut

    # parse args
    global SLIDE_ROOT
    dataset_root = args.dataset_root
    dataset_list = args.dataset_list
    SLIDE_ROOT = args.slide_root

    if not os.path.isdir(dataset_root):
        raise ValueError(f'Dataset root: {dataset_root} is not a directory')
    logger.info(f'Dataset root: {dataset_root}')

    if not os.path.isdir(SLIDE_ROOT):
        raise ValueError(f'WSI root: {SLIDE_ROOT} is not a directory')

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