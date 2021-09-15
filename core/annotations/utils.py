# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: TCTPreprocess
@File: utils.py.py
@Date: 2021/9/10 
@Time: 上午10:46
@Desc: Tools for processing annotations.
'''

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