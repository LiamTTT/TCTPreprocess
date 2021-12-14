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


def check_file(file_path, raise_error=False, logger=None, mute=True):
    existed = os.path.isfile(file_path)
    if not existed:
        msg = f"{file_path} is not existed!"
        if raise_error:
            raise FileNotFoundError(msg)
        elif logger is not None:
            logger.warning(msg)
        else:
            if not mute:
                print(msg)
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
    try:
        im = Image.open(image_path)
        im.verify()  # PIL verify
    except Exception as exc:
        return False
    with open(image_path, 'rb') as f:
        f.seek(-2, 2)
        return f.read() == b'\xff\xd9'


def verify_png(image_path):
    """ inspired by https://github.com/ultralytics/yolov5/issues/916#issuecomment-862208988
        Check whether the PNG is corrupted PNG.
    """
    try:
        im = Image.open(image_path)
        im.verify()  # PIL verify
    except Exception as exc:
        return False
    with open(image_path, 'rb') as f:
        f.seek(-8, 2)
        return f.read() == b'IEND\xaeB`\x82'


def match_sld_in_list(sld_n, sld_list):
    nb_sld = len(sld_list)
    sl1 = [sn.split(' ')[0] for sn in sld_list]
    sl2 = [sn.split('_')[0] for sn in sld_list]
    sl3 = ['_'.join(sn.split('_')[1:]) for sn in sld_list if len(sn.split('_')) > 1]
    sl4 = ['_'.join(sn.split(' ')[1:]) for sn in sld_list if len(sn.split(' ')) > 1]
    sl5 = [sn.split('_')[1] for sn in sld_list if len(sn.split('_')) > 1]
    sl6 = [' '.join(sn.split(' ')[1:]) for sn in sld_list if len(sn.split(' ')) > 1]
    sl7 = [' '.join(sn.split('_')[1:]) for sn in sld_list if len(sn.split('_')) > 1]
    sl8 = [sn.split(' ')[1] for sn in sld_list if len(sn.split(' ')) > 1]
    sl = sl1 + sl2 + sl2 + sl3 + sl4 + sl5 + sl6 + sl7 + sl8
    if sld_n in sld_list:
        return sld_n
    else:
        if sld_n in sl:
            s_id = sl.index(sld_n) % nb_sld
            matched_sld = sld_list[s_id]
            return matched_sld
        elif sld_n.split(' ')[0] in sl:
            s_id = sl.index(sld_n.split(' ')[0]) % nb_sld
            matched_sld = sld_list[s_id]
            return matched_sld
        elif sld_n.split('_')[0] in sl:
            s_id = sl.index(sld_n.split('_')[0]) % nb_sld
            matched_sld = sld_list[s_id]
            return matched_sld
        elif '_'.join(sld_n.split('_')[1:]) in sl:
            s_id = sl.index('_'.join(sld_n.split('_')[1:])) % nb_sld
            matched_sld = sld_list[s_id]
            return matched_sld
        else:
            return None



