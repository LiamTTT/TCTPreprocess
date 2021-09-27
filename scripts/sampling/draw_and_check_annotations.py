# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: draw_and_check_annotations.py.py
@Date: 2021/9/22 
@Time: 下午12:17
@Desc: script for drawing annotations on images for checking
'''
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import xmltodict
from loguru import logger

from core.common import check_dir


def process_annotaion(annotation_path, image_path, save_path):
    annotation = xmltodict.parse(open(annotation_path, 'rb'))['annotation']
    det_objects = annotation['object']
    if not isinstance(det_objects, list):
        det_objects = [det_objects]
    try:
        img = cv2.imread(image_path)
        color = (0, 0, 255)
        thickness = 5
        for obj in det_objects:
            bndbox = obj['bndbox']
            pt1 = (int(float(bndbox['xmin'])), int(float(bndbox['ymin'])))
            pt2 = (int(float(bndbox['xmax'])), int(float(bndbox['ymax'])))
            cv2.rectangle(img, pt1, pt2, color, thickness)
        return cv2.imwrite(save_path, img)
    except Exception as exc:
        logger.exception(exc)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drawing annotations on images for checking")
    parser.add_argument('-r', '--dataset_root', type=str, help="Diractory to dataset.")
    parser.add_argument('-s', '--save_root', type=str, default=None, help='save root')
    args = parser.parse_args()

    dataset_root = args.dataset_root
    save_root = args.save_root
    if save_root is None: save_root = dataset_root
    images_dir = os.path.join(dataset_root, 'JPEGImages')
    annotations_dir = os.path.join(dataset_root, 'Annotations')
    save_dir = os.path.join(save_root, 'checking')
    check_dir(save_dir, True, logger)

    image_names = [f.split('.xml')[0] for f in os.listdir(annotations_dir)]
    nb_success = 0
    time_fmt = "%a %b %d %H:%M:%S %Y"
    logger.info(f'processing {os.path.basename(dataset_root)} start time: {time.strftime(time_fmt, time.localtime())}')
    with ThreadPoolExecutor(5, f'{os.path.basename(dataset_root)}-ThreadPool') as executor:
        futures_draw = {}
        for img_n in image_names:
            annotation_path = os.path.join(annotations_dir, img_n + '.xml')
            image_path = os.path.join(images_dir, img_n + '.jpg')
            save_path = os.path.join(save_dir, img_n + '.jpg')
            futures_draw[executor.submit(process_annotaion, annotation_path, image_path, save_path)] = image_path
        for future_draw in futures_draw:
            image_path = futures_draw[future_draw]
            try:
                res = future_draw.result()
                if not res:
                    logger.error(f'{image_path} failed')
                    continue
                nb_success += 1
            except Exception as exc:
                logger.exception(f'{image_path} failed')
    logger.info(f'finish {nb_success}/{len(image_names)} time: {time.strftime(time_fmt, time.localtime())}')
