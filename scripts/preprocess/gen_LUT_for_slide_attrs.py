# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Project: TCTPreprocess
@File: gen_LUT_for_slide_attrs.py
@Date: 2021/12/1308
@Time: 上午11:43
@Desc: Generating attributes LUT for WSIs.
P.S. The getting attributes function of SDPC API in linux could not work. Running this script in Windows.
'''
import os
import json
import argparse

from loguru import logger
from tqdm import tqdm
from openslide import OpenSlideError

from core.slide_reader import SlideReader

handle = SlideReader()

# *** FCDD dataset should notice that ***
# at the first time, the scripts was designed for FCDD dataset.
# now a general version is applied, the format of files below may not correct.


def is_bat_wsi(wsi_path, suffix):
    return os.path.isfile(wsi_path) and wsi_path.split('.')[-1] == suffix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate Attr LUT file for linux.")
    parser.add_argument("-f", "--batch2x", type=str, default="Batch2X.json",
                        help="path to batch to x file, which should contain keys: BAT_TO_RELATED_DIR, BAT_TO_ROOT and "
                             "BAT_TO_SUFFIX.")
    parser.add_argument("-s", "--save_path", type=str, help="path to save LUT file.")
    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    print(f'save in {save_path}')

    with open(args.batch2x, 'r') as f:
        batch2x = json.load(f)
    bat2dir = batch2x["BAT_TO_RELATED_DIR"]
    bat2root = batch2x["BAT_TO_ROOT"]
    bat2suffix = batch2x["BAT_TO_SUFFIX"]

    total_attrs = {}
    for batch_n, batch_d in bat2dir.items():
        sld_root = bat2root[batch_n]
        batch_suffix = bat2suffix[batch_n]

        batch_dir = os.path.join(sld_root, batch_d)
        slide_list = [s for s in os.listdir(batch_dir) if is_bat_wsi(os.path.join(batch_dir, s), batch_suffix)]

        logger.info(f'{batch_n} {batch_dir}')
        total_attrs[batch_n] = {}
        for sld_n in tqdm(slide_list):
            sld_p = os.path.join(batch_dir, sld_n)
            try:
                handle.open(sld_p)
                sld_attrs = handle.get_attrs()
                total_attrs[batch_n][sld_n] = sld_attrs
            except OpenSlideError as ose:
                logger.error(f'{batch_n} {sld_n} {sld_p} {str(ose)}')
                continue
            except:
                logger.error(f'{batch_n} {sld_n} {sld_p} get_attrs() error!')
                continue
            handle.close()
        logger.info(f'finish {batch_n}')
        logger.info('Original slide num: {:d}'.format(len(slide_list)))
        logger.info('Actual slide num: {:d}'.format(len(total_attrs[batch_n])))
    json.dump(total_attrs, open(save_path, 'w+'), indent=2)
