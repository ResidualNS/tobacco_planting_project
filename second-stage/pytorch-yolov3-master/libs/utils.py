from lxml import etree
import math
import os
from libs.pascal_voc_io import PascalVocWriter


def str2num(str_):
    shape_points = [tuple(map(float, p.split(','))) for p in str_.split(';')]
    points = []
    for temp_p in shape_points:
        points.append(temp_p[0])
        points.append(temp_p[1])
    return points


def parse_anno_file(cvat_xml):
    """读取标注的有问题的矩形框的xml"""
    root = etree.parse(cvat_xml).getroot()
    anno = []
    for image_tag in root.iter('image'):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['polygon'] = []
        image["box"] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['polygon'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            # box = []
            for key, value in box_tag.items():
                box[key] = value
            # box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
            #     box['xtl'], box['ytl'], box['xbr'], box['ybr'])

            new_box = [int(float(box["xtl"])), math.ceil(float(box["ytl"])),
                       int(float(box["xbr"])), math.ceil(float(box["ybr"])),
                       box["label"]]
            image['box'].append(new_box)

        image['polygon'].sort(key=lambda x: int(x.get('z_order', 0)))
        # image['box'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)

    return anno


def save_xml(imgFolderName, imgFileName, imagePath, imageShape, bndbox_list, save_xml_path, difficult=0):
    # imgFolderName = 'test'
    # imgFileName = '2007_000032.jpg'
    # imagePath = '/media/workspaces/test/2007_000032.jpg'
    # imageShape = [375, 500, 3]
    writer = PascalVocWriter(foldername=imgFolderName, filename=imgFileName, imgSize=imageShape,
                             localImgPath=imagePath)
    # difficult = 0
    for bndbox_label in bndbox_list:
        bndbox = bndbox_label[:4]
        label = bndbox_label[-1]
        # bndbox = (94, 178, 215, 231)
        # label = 'dog'
        writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult)  # 添加框及坐标及困难度
    filename = os.path.join(save_xml_path, imgFileName[:-4] + '.xml')
    writer.save(targetFile=filename)
