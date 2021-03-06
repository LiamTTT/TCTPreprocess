# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Project: TCTPreprocess
@File: cvt2voc_FCDD.py
@Date: 2021/9/10 
@Time: 上午10:55
@Desc: convert FCDD xml annotation to voc format with specified hyper params.
'''
import json
import os
import copy
import argparse
import random
from time import time

import xmltodict
import numpy as np
from loguru import logger
from glob2 import glob
from tqdm import tqdm

from core.annotations import (
    create_voc_annotation, create_voc_object,
    find_neighbors_for_given_center, get_bnd_in_tile,
    BndBox
)
from core.common import save_xml_str, check_dir
from core.slide_reader import SlideReader

DST_MPP = 0.67  # um/pixel Portable Microscope
random.seed(42)


def format_det_object_dict(det_obj_dict):
    """ convert str to appropriate data type
    :param det_obj_dict: object gotten from origin xml file
    :return: a formatted object dict.
    """
    det_obj_dict = copy.deepcopy(det_obj_dict)
    det_obj_dict['truncated'] = bool(int(det_obj_dict['truncated']))
    det_obj_dict['difficult'] = bool(int(det_obj_dict['difficult']))
    det_obj_dict['bndbox']['xmin'] = int(float(det_obj_dict['bndbox']['xmin']))
    det_obj_dict['bndbox']['ymin'] = int(float(det_obj_dict['bndbox']['ymin']))
    det_obj_dict['bndbox']['xmax'] = int(float(det_obj_dict['bndbox']['xmax']))
    det_obj_dict['bndbox']['ymax'] = int(float(det_obj_dict['bndbox']['ymax']))
    return det_obj_dict


def get_obj_in_tile(det_object, center_delta, bnd_size, tile_size):
    """ Create objects in tile
    :param det_object: the origin object in WSI, for getting necessary info for creating new object.
    :param center_delta: the delta in x and y of current object from center of the tile
    :param bnd_size: the bounding width and height of origin object.
    :param tile_size: tile size
    :return: new object, segmented
    """
    obj_kwargs = copy.deepcopy(det_object)
    obj_kwargs = {**obj_kwargs, **obj_kwargs['bndbox']}
    obj_kwargs.pop('bndbox')
    (obj_kwargs['xmin'], obj_kwargs['ymin'], obj_kwargs['xmax'], obj_kwargs['ymax']), seg = get_bnd_in_tile(center_delta,
                                                                                                            bnd_size,
                                                                                                            tile_size)
    if obj_kwargs['xmin'] == obj_kwargs['xmax'] or obj_kwargs['ymin'] == obj_kwargs['ymax']:
        # FIXME 211217 siboliu: The main reason for zero obj is not clear, this condition is temporal solution.
        return None, None
    return create_voc_object(**obj_kwargs), seg


def create_annotation_of_slide(xml_path, tile_size, center_anno=True, nb_sample=1):
    """ Create all annotations in the slide
    :param xml_path: FCDD annotation xml file for WSIs
    :param tile_size: specified tile size
    :return: list of annotation objects (could be empty).
    """
    annotations = []
    # parse xml
    xml_content = xmltodict.parse(open(xml_path, 'rb'))
    if 'object' not in xml_content['annotation'].keys():  # check if there is any object to process
        logger.warning(f'No object in file: {xml_path}')
        return annotations
    # Meta info for all the annotations in the xml file.
    xml_filename = xml_content['annotation']['filename']
    image_device = 'MicroScope'
    wsi_batch = xml_filename.split('_')[0]  # e.g. SFY1P
    wsi_suffix = BAT_TO_SUFFIX[wsi_batch]  # e.g. mrxs
    wsi_name = xml_filename.split(wsi_batch+'_')[-1] + f'.{wsi_suffix}'  # xxx.xx  e.g. 01.mrxs
    folder = wsi_batch
    # get slide attrs
    wsi_attrs = ATTRS_LUT[wsi_batch][wsi_name]  # attributes: mpp level width height bound_init level_ratio
    # convert tile size to wsi mpp
    tile_size = (np.array(tile_size)*DST_MPP/wsi_attrs['mpp']).astype(int)
    # process object
    xml_objects = xml_content['annotation']['object']
    if not isinstance(xml_objects, list):
        xml_objects = [xml_objects]
    det_objects = [format_det_object_dict(ob) for ob in xml_objects]  # convert str to appropriate data type
    bounding_boxes = np.array([[ob['bndbox']['xmin'], ob['bndbox']['ymin'], ob['bndbox']['xmax'], ob['bndbox']['ymax']] for ob in det_objects])
    bnds_center = (bounding_boxes[:, :2] + bounding_boxes[:, 2:])//2
    bnds_wh = bounding_boxes[:, 2:] - bounding_boxes[:, :2] + 1
    # create xml file
    for idx, det_object in enumerate(det_objects):
        # get main object
        main_shifts = []
        for shift_id in range(nb_sample):
            shift_x = (random.random() - 0.5) * 0.90 * tile_size[0]
            shift_y = (random.random() - 0.5) * 0.90 * tile_size[1]
            main_shifts.append(np.array([shift_x, shift_y], dtype=int))
        if center_anno:
            main_shifts[0] = np.zeros((2, ))
        for shift_id, main_shift in enumerate(main_shifts):
            objects = []
            obj_in_tile, segmented = get_obj_in_tile(det_object, main_shift, bnds_wh[idx], tile_size)
            if obj_in_tile is None:
                logger.warning(f"annotations is invalid, skip: {xml_filename}, {idx} annotation, shift id: {shift_id}, bnd: {bnds_wh[idx]}")
                break
            objects.append(obj_in_tile)
            neighbor_idx, neighbor_delta = find_neighbors_for_given_center(idx, main_shift, tile_size, bnds_center)
            for ni, nd in zip(neighbor_idx, neighbor_delta):
                obj_in_tile, seg = get_obj_in_tile(det_objects[ni], main_shift + nd, bnds_wh[ni], tile_size)
                if obj_in_tile is None:
                    continue
                segmented |= seg
                objects.append(obj_in_tile)
            # define other attributes in annotation file.
            filename = '{:s}_{:0>3d}_{:d}_{:d}_{:d}_{:s}.jpg'.format(wsi_name.split(f'.{wsi_suffix}')[0], idx, shift_id, *bnds_center[idx], det_object['name'])
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
                segmented
            )
            annotations.append(voc_anno)
    return annotations


def create_annotation_of_batch(xml_dir, tile_size, center_anno=True, nb_sample=1):
    """ Create all annotations in a batch of WSIs
    :param xml_dir: Dir to FCDD annotation xml files for a batch of WSIs
    :param tile_size: tile size
    :return: list of annotations(could be empty.)
    """
    logger.info(f'Process {xml_dir}')
    xmls_path = glob(os.path.join(xml_dir, '*.xml'))
    logger.info(f'File nums: {len(xmls_path)}')
    nb_success = 0
    batch_annotations = []
    since = time()
    for xml_path in tqdm(xmls_path, desc=f'Process {xml_dir}'):
        try:
            slide_annotations = create_annotation_of_slide(xml_path, tile_size, center_anno=center_anno, nb_sample=nb_sample)
        except:
            logger.exception(f'process {xml_path} failed!')
            continue
        batch_annotations += slide_annotations
        nb_success += 1
    logger.info(f'{nb_success} files success!')
    logger.info(f'Total annotations: {len(batch_annotations)}')
    total_time = time() - since
    logger.info('Time: {:.2f} s, {:.3f} s/slide'.format(total_time, total_time/nb_success))
    return batch_annotations


def create_dataset_for_batch(dataset_root, xml_dir, tile_size, center_anno=True, nb_sample=1):
    """ Create and save annotations file for a dataset.
    :param dataset_root: dataset root for saving annotations
    :param xml_dir: related FCDD annotation xml files
    :param tile_size: tile size
    """
    logger.info(f'Create dataset Annotations file: {dataset_root}')
    save_annos_dir = os.path.join(dataset_root, 'Annotations')
    check_dir(save_annos_dir, True, logger)
    batch_annotations = create_annotation_of_batch(xml_dir, tile_size, center_anno=center_anno, nb_sample=nb_sample)
    nb_success = 0
    since = time()
    for anno in tqdm(batch_annotations, desc=f'Save annotations in {save_annos_dir}'):
        save_path = os.path.join(save_annos_dir, anno.filename.replace('jpg', 'xml'))
        if save_xml_str(anno.to_xml(), save_path, 'w+', logger):
            nb_success += 1
    total_time = time() - since
    logger.info(f'create {nb_success} annotations file in {save_annos_dir}')
    logger.info('Time: {:.2f} s {:.3f} s/file'.format(total_time, total_time/nb_success))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert FCDD xml to VOC annotation files.')
    parser.add_argument('--xml_root', help='dir to FCDD xml files', type=str)
    parser.add_argument('--dataset_root', help='dir to all FCDD dataset root', type=str)
    parser.add_argument('--dataset_list', help='dataset name list, also the folder name. Empty for all',
                        type=str, nargs='*', default=[])
    parser.add_argument('--tile_size', help='tile size in 0.67um/pixel',
                        type=int, nargs='+', default=[800])
    parser.add_argument('--center_anno', default=False, action='store_true', help='always create center annotation')
    parser.add_argument('--nb_sample', default=1, type=int, help='number of sample to create.')
    parser.add_argument('--slide_root', help='dir to WSI files', type=str)
    parser.add_argument('--batch_to_x', help='path to batch to x file', type=str)
    parser.add_argument('--attrs_lut', help='path to WSI attributes', type=str)
    args = parser.parse_args()

    # parse lut
    global BAT_TO_DIR
    global BAT_TO_SUFFIX
    global ATTRS_LUT
    BAT_TO_X = json.load(open(args.batch_to_x, 'r'))
    BAT_TO_DIR = BAT_TO_X['BAT_TO_DIR']
    BAT_TO_SUFFIX = BAT_TO_X['BAT_TO_SUFFIX']
    ATTRS_LUT = json.load(open(args.attrs_lut, 'r'))
    # end parse lut

    # parse rrgs
    xml_root = args.xml_root
    dataset_root = args.dataset_root
    dataset_list = args.dataset_list
    tile_size = args.tile_size
    slide_root = args.slide_root
    center_anno = args.center_anno
    nb_sample = args.nb_sample
    if nb_sample < 1 or nb_sample > 9:
        nb_sample = np.clip(nb_sample, 1, 9)
        logger.warning(f"clip nb_sample to {nb_sample}")
    logger.info(f'create {nb_sample} samples for each annotation. make annotaion at center: {center_anno}')
    if not os.path.isdir(xml_root):
        raise ValueError(f'FCDD xml root: {slide_root} is not a directory')
    logger.info(f'FCDD xml root: {xml_root}')

    check_dir(dataset_root, True, logger)
    if not os.path.isdir(dataset_root):
        raise ValueError(f'Dataset root: {dataset_root} is not a directory')
    logger.info(f'Dataset root: {dataset_root}')

    if len(tile_size) == 1:
        tile_size = tile_size * 2
    elif len(tile_size) == 2:
        pass
    else:
        raise ValueError(f'Tile size {tile_size} should be [width, height] or [length] for square')

    if not os.path.isdir(slide_root):
        raise ValueError(f'WSI root: {slide_root} is not a directory')

    if len(dataset_list) == 0:
        dataset_list = [fd for fd in os.listdir(xml_root) if os.path.isdir(os.path.join(xml_root, fd))]
        logger.info(f'Create all dataset in {xml_root}')
    logger.info(f'Datasets: {dataset_list}')
    logger.info(f'Tile Size: {tile_size}')
    # end parse args

    # process
    for dataset_name in dataset_list:
        dataset_dir = os.path.join(dataset_root, dataset_name)
        check_dir(dataset_dir, True, logger)
        xml_dir = os.path.join(xml_root, dataset_name)
        create_dataset_for_batch(dataset_dir, xml_dir, tile_size, center_anno=center_anno, nb_sample=nb_sample)
    # end process
