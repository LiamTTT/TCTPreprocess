# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: merge_dataset.py
@Date: 2021/10/12 
@Time: 下午4:21
@Desc: Merging datasets.
'''

import os
import argparse
from time import time
from shutil import copyfile
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, as_completed)

from loguru import logger

from core.common import check_dir

SUBSETS = ['train', 'val', 'test', 'trainval']


def copy_anno_and_img(src_path, dst_path):
    anno_format = ('Annotations', 'xml')
    jpeg_format = ('JPEGImages', 'jpg')
    try:
        copyfile(src_path % anno_format, dst_path % anno_format)
        copyfile(src_path % jpeg_format, dst_path % jpeg_format)
    except Exception as exc:
        logger.exception(f"copy file error: {src_path} to {dst_path}")
        return False
    return True


def merge(dataset_root, datasets, subset, new_dir):
    # create folders
    anno_dir = os.path.join(new_dir, 'Annotations')
    jpeg_dir = os.path.join(new_dir, 'JPEGImages')
    sbst_dir = os.path.join(new_dir, 'ImageSets', 'Main')
    check_dir(anno_dir, True)
    check_dir(jpeg_dir, True)
    check_dir(sbst_dir, True)
    # merge txt and record file to move
    src_files = []
    dst_files = []
    for sbst in subset:
        merge_lines = []
        sbst_path = os.path.join(sbst_dir, sbst + '.txt')
        for ds in datasets:
            cur_ds_root = os.path.join(dataset_root, ds)
            cur_ds_sbst_dir = os.path.join(cur_ds_root, 'ImageSets', 'Main')
            # merge txt
            with open(os.path.join(cur_ds_sbst_dir, sbst + '.txt'), 'r') as f:
                cur_lines = f.readlines()
            merge_lines += cur_lines
            # record file path for copying
            for l in cur_lines:
                name = l.strip()
                src_path = os.path.join(cur_ds_root, "%s", name + ".%s")
                dst_path = os.path.join(new_dir, "%s", name + ".%s")
                src_files.append(src_path)
                dst_files.append(dst_path)
        # write merged txt
        with open(sbst_path, 'w+') as f:
            f.writelines(merge_lines)
        logger.info(f"write {sbst} set to {sbst_path}")
    logger.info(f"copy files, total num: {len(src_files)} ...")
    nb_success = 0
    since = time()
    with ThreadPoolExecutor(8) as executor:
        future_to_copy = {
            executor.submit(copy_anno_and_img, *p): p for p in zip(src_files, dst_files)
        }
        for future in future_to_copy:
            p = future_to_copy[future]
            res = future.result()
            if res:
                nb_success += 1
            else:
                logger.error("error: %s to %s" % p)
    logger.info(f'success {nb_success}')
    logger.info(f'time consuming: {(time() - since) / 60} mins')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge datasets.')
    parser.add_argument('-r', '--dataset_root', help='dir to all FCDD dataset root', type=str)
    parser.add_argument('-d', '--datasets', help='datasets to merge', type=str, nargs='*', default=[])
    parser.add_argument('-s', '--subset', help='subsets(train test val or trainval) to merge', type=str,
                        nargs='*', default=[])
    parser.add_argument('-n', '--new_name', help='name of new dataset', type=str)
    parser.add_argument('-nr', '--new_root', help='root to save new dataset', type=str,
                        default=None)
    args = parser.parse_args()

    # parse args
    dataset_root = os.path.abspath(args.dataset_root)
    datasets = args.datasets
    subset = args.subset
    new_name = args.new_name
    new_root = args.new_root
    if new_root is None:
        new_root = dataset_root
    new_root = os.path.abspath(new_root)

    if len(datasets) == 0:
        datasets = [
            fd for fd in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, fd)) and fd.isupper()
        ]
        logger.info(f'Merge all dataset in {dataset_root}')
    logger.info(f'Datasets: {datasets}')

    if len(subset) == 0:
        subset = SUBSETS
    logger.info(f'Subset: {subset}')

    new_dir = os.path.join(new_root, new_name)
    check_dir(new_dir, True, logger)
    logger.info(f'save new dataset in {new_dir}')
    # end parse args

    # process
    # merge ImageSets
    merge(dataset_root, datasets, subset, new_dir)