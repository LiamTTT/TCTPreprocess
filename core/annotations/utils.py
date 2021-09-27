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
