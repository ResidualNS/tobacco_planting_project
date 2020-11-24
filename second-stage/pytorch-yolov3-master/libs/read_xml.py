#!/usr/bin/python
# -*- coding: UTF-8 -*-
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET


def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    bbox_list = []
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        bbox_list.append([x1, y1, x2, y2, ObjName])
    print(bbox_list)
    return bbox_list

if __name__ == '__main__':
    xml_path = '/media/workspaces/biaozhu/write_xml/0723_xml/DJI_0447_json_1_1.xml'
    # read_file(xml_path)
    GetAnnotBoxLoc(xml_path)

