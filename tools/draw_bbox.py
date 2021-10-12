# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: draw_bbox.py
@Date: 2021/9/30 
@Time: 下午4:34
@Desc: Draw bbox for predicting in (name score xmin ymin xmax ymax)
'''
import os
import argparse

import cv2
from loguru import logger

from core.common import check_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="draw bbox in samples")
    parser.add_argument('-r', '--dataset_root', help='dir to specified sub dataset root', type=str)
    parser.add_argument('-b', '--base_fold', help='fold to original images dir in root', type=str, default='JPEGImage')
    parser.add_argument('-f', '--results_file', help='path to predicting results file', type=str)
    # parser.add_argument('-n', '--save_fold_name', help='path to save res images', type=str)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    base_fold = args.base_fold
    results_file = args.results_file

    save_dir = os.path.join(dataset_root, 'result_images')
    check_dir(save_dir, True, logger)

    with open(results_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split(' ') for l in lines]
    lines = [[' '.join(el[:-5])] + el[-5:] for el in lines]
    res_dict = {}
    for el in lines:
        if el[0] not in res_dict.keys():
            res_dict[el[0]] = []
        res_dict[el[0]].append(el[1:])
    logger.info(f'porcessing {dataset_root} bbox num: {len(lines)} samples: {len(res_dict)}')
    logger.info(f'save in {save_dir}')
    for im_n, res in res_dict.items():
        im_p = os.path.join(dataset_root, base_fold, f'{im_n}.jpg')
        im_sp = os.path.join(save_dir, f'{im_n}.jpg')
        try:
            im = cv2.imread(im_p)
        except Exception as exc:
            logger.exception(f'read {im_p} failed')
            continue
        for re in res:
            re = [float(it) for it in re]
            s, xmin, ymin, xmax, ymax = re
            try:
                cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (34,34,178), 5)
                cv2.putText(im, '{:.3f}'.format(s), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (34,34,178), 2)
            except Exception as exc:
                logger.exception(f'draw {re} in {im_p} failed.')
                continue
        cv2.imwrite(im_sp, im)

