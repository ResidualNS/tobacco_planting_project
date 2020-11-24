from models import *
from utils.datasets import *
from utils.utils import *
import Picture_Slicing_Processing as PSP
import draw_toolbox
from detect_copy import param
from Picture_Slicing_Processing import readTif
import cv2
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def write_xml_cvat(xml_path, img_name, height, width, contours_hemiao_rect):
    from lxml import etree

    # 1级目录
    annotation = etree.Element("annotation")

    # 2级目录
    etree.SubElement(annotation, "version").text = "1.1"
    meta = etree.SubElement(annotation, "meta")
    # image_name = img_path.split('\\')[-1]
    image = etree.Element("image", {"id": '0', "name": img_name, "width": str(width), "height": str(height)})
    annotation.append(image)

    # 3级目录
    task = etree.SubElement(meta, "task")
    etree.SubElement(meta, "dumped").text = "2020-06-08 08:22:06.258023+00:00"

    # 4级目录
    etree.SubElement(task, "id").text = "130"
    etree.SubElement(task, "name").text = "task_name"
    etree.SubElement(task, "size").text = "1"
    etree.SubElement(task, "mode").text = "annotation"
    etree.SubElement(task, "overlap").text = "0"
    etree.SubElement(task, "bugtracker")
    etree.SubElement(task, "created").text = "2020-05-20 01:38:39.362995+00:00"
    etree.SubElement(task, "updated").text = "2020-05-20 08:00:34.872446+00:00"
    etree.SubElement(task, "start_frame").text = "0"
    etree.SubElement(task, "stop_frame").text = "0"
    etree.SubElement(task, "frame_filter").text = ' '
    etree.SubElement(task, "z_order").text = "False"

    labels = etree.SubElement(task, "labels")
    # ---包含5,6级目录---
    label = etree.SubElement(labels, "label")
    etree.SubElement(label, "name").text = "yanmiao"

    segments = etree.SubElement(task, "segments")
    # ---包含5,6级目录---
    segment = etree.SubElement(segments, "segment")
    etree.SubElement(segment, "id").text = "112"
    etree.SubElement(segment, "start").text = "0"
    etree.SubElement(segment, "stop").text = "0"
    etree.SubElement(segment, "url").text = "http://10.10.0.120:8080/?id=112"

    owner = etree.SubElement(task, "owner")
    # ---包含5,6级目录---
    etree.SubElement(owner, "username").text = "kefgeo"
    etree.SubElement(owner, "email").text = "kefgeo@kefgeo.com"

    assignee = etree.SubElement(task, "assignee")
    # ---包含5,6级目录---
    etree.SubElement(assignee, "username").text = "kefgeo"
    etree.SubElement(assignee, "email").text = "kefgeo@kefgeo.com"

    # meta结束，开始image
    # 保存box
    dianzhu_num = 0
    for point in contours_hemiao_rect:
        dianzhu_num += 1
        # 保存烟苗识别结果：矩形框
        # dict_info = {'xyxy': xyxy, 'label': label, 'conf': conf}
        # xtl, ytl, xbr, ybr = point['xtl'], point['ytl'], point['xbr'], point['ybr']
        xyxy = point
        xtl, ytl, xbr, ybr = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        #conf = '%.3f' % float(point['conf'])
        xtl, ytl, xbr, ybr = str(xtl), str(ytl), str(xbr), str(ybr)
        box = etree.Element("box", {"label": 'yanmiao', "occluded": '0',
                                    "xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr})
        image.append(box)

    # 2种记录点株方式
    # dianzhu_num = etree.Element("image", {"dianzhu_num": str(dianzhu_num)})
    # image.append(dianzhu_num)
    etree.SubElement(image, "dianzhu_num").text = str(dianzhu_num)
    print('点株数量：', dianzhu_num)

    #print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))

    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    xml_string = ET.tostring(annotation)
    dom = minidom.parseString(xml_string)
    with open(xml_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')

    # print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    # etree.ElementTree(anno_tree).write(save_path, encoding='UTF-8', pretty_print=True)

    print('xml_path:',xml_path)
    print('---------------finish save_xml---------------')



def big_detect():

    labels_total = []
    scores_total = []
    bboxes_total = []

    im = readTif(im_path)
    H=im.shape[0]
    W=im.shape[1]
    sub_img, site = PSP.splitimage(im_path, shape=[416, 416], strided=300)

    num=0
    for image in sub_img:  # 子图
        image_name='./data/samples/'+str(num)+'.png'
        cv2.imwrite(image_name,image)
        labels_, scores_, bboxes_ = param(True,Source=image_name)  # 单张子图结果
        #os.remove(image_name)
        #num_bbox = len(labels_)
        labels_total.append(labels_)
        scores_total.append(scores_)
        bboxes_total.append(bboxes_)
        num +=1
        print('num=',num)

    #print('labels_total:',labels_total)
    #print('scores_total:', scores_total)
    #print('bboxes_total:', bboxes_total)
    print('num of  bbo：', len(bboxes_total))

    labels_merge, scores_merge, bboxes_merge = PSP.merge_label(labels_total, scores_total, bboxes_total, site,
                                                                       [416, 416], im.shape)

    result_img= draw_toolbox.bboxes_draw_on_img(im, labels_merge, scores_merge, bboxes_merge)

    # label_id_dict = draw_toolbox.gain_translate_table()
    # result_num = {}
    # for i in range(len(labels_merge)):
    #     temp = labels_merge[i]
    #     if temp not in result_num.keys():
    #         result_num[temp] = 1
    #     else:
    #         result_num[temp] = result_num[temp] + 1
    # class_num = ''
    # for key, value in result_num.items():
    #     if key in label_id_dict.keys():
    #         class_num = class_num + label_id_dict[key] + "_" + str(value)

    print('--------------- finish detect -----------------------')
    cv2.imwrite(save_path, result_img)
    #np.savetxt(im_path[:-8]+save_name+'_result.txt', bboxes_merge, fmt='%0.3f')
    #write_xml_cvat(xml_path,img_name,H,W,bboxes_merge)
    print('im_path:',im_path)




file_path='./data/testNMS/'
im_path=file_path+'6-7.png'
save_name=im_path.split('/')[-1][:-4]
save_path=file_path+save_name+'_result.png'
xml_path=file_path+save_name+'_result.xml'
img_name=im_path.split('/')[-1]

start_time = time.time()
big_detect()
end_time = time.time()
time_image = end_time - start_time
print('time:',time_image)
print('--------------- finish-----------------------')
