# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: spliting_train_eval_test.py.py
@Date: 2021/9/22 
@Time: 下午4:26
@Desc: Random splitting train eval and test set on slide level
'''
import os
import sys
import argparse
import random
import json
from glob2 import glob

import numpy as np
from loguru import logger

from core.common import check_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting train val test set automatically.')
    parser.add_argument('--dataset_root', type=str,
                        help='root to dataset containing \'Annotations\' and \'JPEGImages\' dir.')
    parser.add_argument('--ratios', type=float, nargs='+', default=[8, 1, 1],
                        help='ratios for trains test val dataset.')
    parser.add_argument('--seed', type=int, default=42, help='random seed.')
    parser.add_argument('--splitting_dict', type=str, help='pre-define splitting dict of slides name.', default=None)
    parser.add_argument('--roi', help="whether contain roi info.", action='store_true')
    args = parser.parse_args()
    SET_NAME = ['train', 'eval', 'test']
    SEED = args.seed
    splitting_dict = args.splitting_dict
    flg_roi = args.roi
    dataset_root = args.dataset_root
    dataset_root = os.path.abspath(dataset_root)
    logger.info(f'Processing {dataset_root}')
    ratios = args.ratios
    if splitting_dict is not None:
        logger.warning(f'Using pre define splitting, the splitting ratio will not be guaranteed: {splitting_dict}')
        splitting_dict = json.load(open(splitting_dict, 'r'))
    if flg_roi:
        logger.warning('Slide is consisted by rois.')
    if not isinstance(ratios, list) or len(ratios) == 0:
        logger.error('arg --ratios get a wrong format!')
        raise ValueError('arg --ratios get a wrong format!')
    if len(ratios) == 1:
        logger.info('average splitting')
        ratios = ratios * 3
    elif len(ratios) == 2:
        ratios = ratios.append(ratios[-1])
    elif len(ratios) == 3:
        pass
    else:
        logger.error('ratios length should be 1, 2 or 3')
        raise ValueError('ratios length should be 1, 2 or 3')
    logger.info('train : evel : test = {} : {} : {}'.format(*ratios))
    ratios = np.array(ratios)

    splitting_file_dir = os.path.join(dataset_root, 'ImageSets', 'Main')
    check_dir(splitting_file_dir, create=True, logger=logger)
    train_file_path = os.path.join(splitting_file_dir, 'train.txt')
    eval_file_path = os.path.join(splitting_file_dir, 'val.txt')
    test_file_path = os.path.join(splitting_file_dir, 'test.txt')
    train_n_eval_file_path = os.path.join(splitting_file_dir, 'trainval.txt')
    slide_json = os.path.join(splitting_file_dir, 'splitting.json')

    annotation_dir = os.path.join(dataset_root, 'Annotations')
    jpegimages_dir = os.path.join(dataset_root, 'JPEGImages')
    e_a = check_dir(annotation_dir)
    e_j = check_dir(jpegimages_dir)
    if not e_a & e_j:
        if not e_a: logger.error(f'{annotation_dir} not exist!')
        if not e_j: logger.error(f'{jpegimages_dir} not exist!')
        raise FileNotFoundError('no annotations and samples.')
    # get total samples and slides
    annotations = os.listdir(annotation_dir)
    if flg_roi:
        slide_name = list(set(['_'.join(an.split('_000_')[0].split('_')[:-1]) for an in annotations if '_000_' in an]))
    else:
        slide_name = list(set([an.split('_000_')[0] for an in annotations if '_000_' in an]))
    # compute numbers of each dataset
    nb_all_samples = len(annotations)
    ls_nb_set = nb_all_samples * ratios / ratios.sum()
    ls_nb_set = ls_nb_set.tolist()
    float_ratio = 0.05
    # splitting
    # pre splitting
    pre_ls_samples = [[], [], []]
    pre_ls_slides = [[], [], []]
    if splitting_dict is not None:
        logger.warning('Do splitting according to pre defined splitting json file. '
                       'Slides not mentioned in pre-defined file will be splitted according to ratios.')
        pre_ls_slides = [splitting_dict[k] for k in SET_NAME]
        for i, sld_ns in enumerate(pre_ls_slides):
            for cur_sld_n in sld_ns:
                if cur_sld_n not in slide_name:
                    continue
                pre_ls_samples[i] += [os.path.basename(f).split('.jpg')[0] for f in
                                      glob(os.path.join(jpegimages_dir, f'{cur_sld_n}_*.jpg'))]
                slide_name.remove(cur_sld_n)
            logger.info(
                'Pre-splitting {} set -- slide num: {} sample num: {}'.format(
                    SET_NAME[i], len(pre_ls_slides[i]), len(pre_ls_samples[i])
                ))
    # random splitting
    random.seed(SEED)  # FIXME 20210923 siboliu: the results is still random.
    ls_samples = [[], [], []]
    ls_slides = [[], [], []]
    # splitting per set
    for i, nb_set in enumerate(ls_nb_set):
        cur_nb_set = len(ls_samples[i])
        sig = 0
        while len(slide_name):
            if sig > 999:
                logger.warning(f'can not get a good solution for this splitting {nb_set}')
                break
            cur_sld_n = random.sample(slide_name, 1)[0]
            cur_samples = [os.path.basename(f).split('.jpg')[0] for f in glob(os.path.join(jpegimages_dir, f'{cur_sld_n}_*.jpg'))]
            cur_samples_nb = len(cur_samples)

            if cur_nb_set + cur_samples_nb < nb_set*(1-float_ratio):
                ls_samples[i] += cur_samples
                cur_nb_set += cur_samples_nb
                ls_slides[i].append(cur_sld_n)
                slide_name.remove(cur_sld_n)
            elif cur_nb_set + cur_samples_nb > nb_set*(1+float_ratio):
                sig += 1
                continue
            else:
                ls_samples[i] += cur_samples
                cur_nb_set += cur_samples_nb
                ls_slides[i].append(cur_sld_n)
                slide_name.remove(cur_sld_n)
                break

        if i == 2 and len(slide_name) != 0:
            # adding slides left into test set
            logger.warning('adding slides left into test set')
            for cur_sld_n in slide_name:
                cur_samples = [os.path.basename(f).split('.jpg')[0] for f in glob(os.path.join(jpegimages_dir, f'{cur_sld_n}_*.jpg'))]
                cur_samples_nb = len(cur_samples)

                ls_samples[i] += cur_samples
                cur_nb_set += cur_samples_nb
                ls_slides[i].append(cur_sld_n)
        # add pre splitting
        ls_samples[i] += pre_ls_samples[i]
        ls_slides[i] += pre_ls_slides[i]
        logger.info('{} set -- slide num: {} sample num: {}'.format(SET_NAME[i], len(ls_slides[i]), len(ls_samples[i])))

    open(train_file_path, 'w+').writelines(sorted([l+'\n' for l in ls_samples[0]]))
    open(eval_file_path, 'w+').writelines(sorted([l+'\n' for l in ls_samples[1]]))
    open(test_file_path, 'w+').writelines(sorted([l+'\n' for l in ls_samples[2]]))
    open(train_n_eval_file_path, 'w+').writelines(sorted([l+'\n' for l in ls_samples[0] + ls_samples[1]]))
    json.dump({k: sorted(v) for k, v in zip(SET_NAME, ls_slides)}, open(slide_json, 'w+'), indent=2)
    logger.info(f'done! results save in {splitting_file_dir}')