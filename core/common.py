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

