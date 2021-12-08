# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Project: TCTPreprocess
@File: utils.py.py
@Date: 2021/9/10 
@Time: 上午10:46
@Desc: Tools for processing annotations.
'''
import numpy as np

from .VOCAnnotation import BndBox, Size, Source, Object, Annotation


def create_voc_annotation(
        folder, filename,
        image_device, wsi_batch, wsi_name, wsi_bndbox,
        width, height,
        objects,
        depth=3,
        segmented=False,
        **kwargs
):
    """ API for create voc annotation.
    """
    if 'source' not in kwargs:
        kwargs['source'] = {}
    if 'annotation' not in kwargs:
        kwargs['annotation'] = {}
    voc_source = Source(image_device, wsi_batch, wsi_name, wsi_bndbox, **kwargs['source'])
    voc_size = Size(width, height, depth)
    voc_objects = objects
    voc_annotation = Annotation(
        folder, filename, voc_source, voc_size, voc_objects, segmented, **kwargs['annotation']
    )
    return voc_annotation


def create_voc_object(
        name,
        xmin, ymin, xmax, ymax,
        pose="Unspecified",
        truncated=False,
        difficult=False,
):
    """ API for create voc object.
    """
    bndbox = BndBox(xmin, ymin, xmax, ymax)
    return Object(name, bndbox, pose, truncated, difficult)


def find_neighbors(bnds_center, tile_size):
    """find neighbors for each bnd given.
    :param bnds_center: all bnd centers
    :param tile_size: tile size
    :return: neighbor id and delta to center bnd
    """
    neighbors = []
    deltas = []
    for idx, bnd_center in enumerate(bnds_center):
        centers_delta = bnds_center - bnd_center
        x_neighbor = np.where(np.abs(centers_delta[:, 0]) < tile_size[0])
        y_neighbor = np.where(np.abs(centers_delta[:, 1]) < tile_size[1])
        neighbor = np.intersect1d(x_neighbor[0].tolist(), y_neighbor[0].tolist()).tolist()
        neighbor.remove(idx)
        neighbors.append(neighbor)
        deltas.append([centers_delta[i] for i in neighbor])
    return neighbors, deltas


def find_neighbors_for_given_center(cur_idx, center_shift, tile_size, bnds_center):
    """ find neighbors for given center and its shift in tile.
    :param cur_idx: idx of given center
    :param center_shift: shift from tile center, top left "-", bottom right "+"
    :param tile_size: tile size
    :param bnds_center: all bnds' center
    :return:
    """
    cur_bnd_center = bnds_center[cur_idx]
    bnds_delta = bnds_center-cur_bnd_center
    lt_t, lt_l = tile_size//2 + center_shift
    lt_b, lt_r = tile_size//2 - center_shift
    x_neighbor = np.intersect1d(np.where(bnds_delta[:, 0] > -lt_l)[0], np.where(bnds_delta[:, 0] < lt_r)[0])
    y_neighbor = np.intersect1d(np.where(bnds_delta[:, 1] > -lt_t)[0], np.where(bnds_delta[:, 1] < lt_b)[0])
    neighbor = np.intersect1d(x_neighbor, y_neighbor).tolist()
    neighbor.remove(cur_idx)
    deltas = [bnds_delta[i] for i in neighbor]
    return neighbor, deltas


def get_bnd_in_tile(center_delta, bnd_size, tile_size):
    """ get bnd coordinates in tile
    :param center_delta: current bnd delta to center bnd
    :param bnd_size: current bnd size
    :param tile_size: tile size
    :return: [xmin, ymin, xmax, ymax], segmented
    """
    bnd_center = tile_size//2 + center_delta
    bnd_size = bnd_size//2
    bnd_tl = bnd_center - bnd_size
    bnd_br = bnd_center + bnd_size
    bnd = np.stack((bnd_tl, bnd_br))
    if (bnd < 0).any() or (bnd[:, 0] > tile_size[0]).any() or (bnd[:, 1] > tile_size[1]).any():
        return np.clip(bnd, [0, 0], tile_size).ravel(), True
    else:
        return bnd.ravel(), False


def process_csv_file(csv_file, bmp_info_lut, logger):
    annotations = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]
        logger.info(f'Num of annotations: {len(lines)}')
        nb_success = 0
        for line in lines:
            bat, img_roi, cx, cy, w, h, anno_n, sld_n, scx, scy, roi_p = line.strip().split(',')
            # rename batch name
            try:
                if bat.upper() not in bmp_info_lut[img_roi]['batch']:
                    logger.error(
                        'slide batch({}) does not match to json file({}).'.format(bat, bmp_info_lut[img_roi]['batch']))
                    continue
                bat = bmp_info_lut[img_roi]['batch']
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
