# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: sampling.py
@Date: 2021/9/17 
@Time: 上午9:53
@Desc: sampling according to voc annotations xml file.
'''
import cv2
import numpy as np


def sampling(bnd, source, src_type):
    """ Sampling tile from source. The source can be a ndarray or a SlideReader handle.
    :param bnd: np.array([xmin, ymin , xmax, ymax])
    :param source: source image, can be a roi as ndarray or a wsi as a SlideReader handle.
    :param src_type:
    :return: a ndarray representing a BGR image.
    """
    # TODO 20210917 LiamTTT: need a way to make this function parallel.
    bnd_xy = bnd[:2]
    bnd_wh = bnd[2:] - bnd[:2] + 1
    sample = np.zeros(bnd_wh.tolist()[::-1] + [3], dtype=np.uint8)  # create a container for image
    # compute shift and real crop wh, only top-left
    crop_shift = np.clip(0 - bnd_xy, (0, 0), None)
    crop_xy = np.clip(bnd_xy, (0, 0), None)
    crop_wh = bnd_wh - crop_shift
    #
    if src_type == 'MicroScope':
        image = sampling_wsi_handle(crop_xy, crop_wh, source)
    else:  # PortableMicroScope
        image = sampling_roi_array(crop_xy, crop_wh, source)
    sample[crop_shift[1]:, crop_shift[0]:, ...] = image
    return sample


def sampling_wsi_handle(bnd_xy, bnd_wh, source, level=0):
    image = source.get_tile(tuple(bnd_xy.astype(int).tolist()), tuple(bnd_wh.astype(int).tolist()), level)
    if image is None:
        # When top-left coordinate exceeds bottom or right, mrxs
        # and svs will return a black image in ndarray, however
        # sdpc will return a NoneType.
        # This code will unify the return value to ndarray.
        image = np.zeros(bnd_wh.tolist()[::-1] + [3])
    return image[..., ::-1]


def sampling_roi_array(bnd_xy, bnd_wh, source):
    bnd_xy_end = bnd_xy + bnd_wh
    image = source[bnd_xy[1]: bnd_xy_end[1], bnd_xy[0]: bnd_xy_end[1], ...]
    return image