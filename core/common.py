# -*- coding:utf-8 -*-
'''
@Author: LiamTTT
@Email: nirvanalautt@gmail.com
@Project: TCTPreprocess
@File: common.py
@Date: 2021/9/15 
@Time: 上午10:52
@Desc: common functions.
'''
import os
from xml.dom.minidom import parseString

from PIL import Image


def check_dir(dir, create=False, logger=None):
    existed = os.path.exists(dir)
    if not existed and create:
        if logger is not None:
            logger.debug(f'create {dir}')
        existed = True if os.makedirs(dir) is None else False
    return existed


def save_xml_str(xml_str, save_path, mode='w+', logger=None):
    """Save xml string to xml file.
    """
    dom = parseString(xml_str)
    try:
        with open(save_path, mode) as f:
            dom.writexml(f)
        return True
    except:
        if logger is not None:
            logger.exception(f'Write {save_path} failed! mode: {mode}')
        return False


def verify_jpeg(image_path):
    """ inspired by https://github.com/ultralytics/yolov5/issues/916#issuecomment-862208988
        Check whether the jpeg is corrupted JPEG.
    """
    im = Image.open(image_path)
    try:
        im.verify()  # PIL verify
    except Exception as exc:
        return False
    with open(image_path, 'rb') as f:
        f.seek(-2, 2)
        return f.read() == b'\xff\xd9'