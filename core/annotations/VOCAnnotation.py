# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Project: TCTPreprocess
@File: VOCAnnotation.py
@Date: 2021/9/9 
@Time: 下午3:15
@Desc: Define VOC Annotation Data Structure, and formatting VOC Annotations.
'''
import abc
import copy
from xml.dom.minidom import parseString

import dicttoxml


class VOCElem(abc.ABC):
    def to_dict(self):
        return self.__dict__

    def to_xml(self, root=False, attr_type=False, **kwargs):
        return dicttoxml.dicttoxml(self.to_dict(), root=root, attr_type=attr_type, **kwargs).decode()


class Size(VOCElem):
    """Image Size
    """
    def __init__(self, width: int, height: int, depth=3):
        self.width = width
        self.height = height
        self.depth = depth


class BndBox(VOCElem):
    """Bounding box
    """
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class Source(VOCElem):
    """Recording where the images come from.
    """
    def __init__(self,
                 image_device: str,
                 wsi_batch: str,
                 wsi_name: str,
                 wsi_bndbox: BndBox,
                 **kwargs
                 ):
        """
        :param image_device: Imaging device
        :param wsi_batch: batch of WSI
        :param wsi_name: name of wsi
        :param wsi_bndbox: bounding box in WSI
        :param kwargs: for extension
        """
        self.image_device = image_device
        self.wsi_batch = wsi_batch
        self.wsi_name = wsi_name
        self.wsi_bndbox = wsi_bndbox.to_dict()
        if self.image_device == 'PortableMicroScope' and 'image_roi' not in kwargs.keys():
            kwargs['image_roi'] = ''
        for k, v in kwargs.items():
            self.__setattr__(k, v)


class Object(VOCElem):
    """Object in Image
    """
    def __init__(self,
                 name: str,
                 bndbox: BndBox,
                 pose="Unspecified",
                 truncated=False,
                 difficult=False,
                 ):
        self.name = name
        self.bndbox = bndbox.to_dict()
        self.pose = pose
        self.truncated = int(truncated)
        self.difficult = int(difficult)


class Annotation(VOCElem):
    """Annotation of a image.
    """
    def __init__(self,
                 folder: str,
                 filename: str,
                 source: Source,
                 size: Size,
                 objects: [Object],
                 segmented=False,
                 **kwargs
                 ):
        """Initialize Annotation info
        :param folder: image folder
        :param filename: image filename
        :param size: image size, Size object
        :param objects: list of objects in image
        :param segmented: boolean, whether consist segmented label
        :param source: where the image comes from.
        :param kwargs: for extension.
        """
        self.folder = folder
        self.filename = filename
        self.source = source.to_dict()
        self.size = size.to_dict()
        self.segmented = int(segmented)
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.objects = [obj.to_dict() for obj in objects]

    def to_xml(self):
        attr_dict = copy.deepcopy(self.to_dict())
        objs = attr_dict['objects']
        attr_dict.pop('objects')
        xml_str = dicttoxml.dicttoxml(attr_dict, root=True, custom_root='annotation', attr_type=False).decode()
        if len(objs):
            obj_str = dicttoxml.dicttoxml(objs, root=False, item_func=lambda x: 'object', attr_type=False).decode()
            insert_idx = xml_str.find('</annotation>')
            xml_str = xml_str[:insert_idx] + obj_str + xml_str[insert_idx:]
        return xml_str

