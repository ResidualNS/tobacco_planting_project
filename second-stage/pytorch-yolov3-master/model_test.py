#!/usr/bin/env python
### https://blog.csdn.net/weixin_42111393/article/details/82940681
## 将tf-faster-r cnn中的框画在一张图内
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import warnings
warnings.filterwarnings('ignore')


import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os.path as osp
import os

from Crop_Detection.lib.config import config as cfg
from Crop_Detection.lib.utils.nms_wrapper import nms
from Crop_Detection.lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from Crop_Detection.lib.nets.vgg16 import vgg16
from Crop_Detection.lib.utils.timer import Timer

import Picture_Slicing_Processing as PSP
import draw_toolbox as draw_toolbox
import dataset_common  as  dataset_common

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__','bleeding_black','bleeding_leaf','bleeding_leaf_big','bleeding_flower')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_68500.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('default',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def get_bbox( class_name, im, dets, thresh=0.5):
    # cls_bbox_score = {}
    all_bbox = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return all_bbox
    else:
        print('Box numbers: ', class_name, ' ', len(inds))
        h, w, _ = im.shape
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            # temp_bbox_yx = []
            # temp_bbox_yx.append(bbox[1]/h)  # y1
            # temp_bbox_yx.append(bbox[0] / w)  # x1
            # temp_bbox_yx.append(bbox[3] / h)   # y2
            # temp_bbox_yx.append(bbox[2] / w)  # x2
            # temp_bbox_yx.append(score)  # 得分
            # all_bbox.append(temp_bbox_yx)
            all_bbox.append(dets[i])
        # if len(all_bbox) != 0:
        #     cls_bbox_score[class_name] = all_bbox
        return all_bbox


def vis_detections(ax, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    print('Box numbers: ', class_name, ' ',len(inds))
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print(bbox[0], bbox[1], bbox[2], bbox[3])
        ## 红框
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1)## linewidth=3.5
        )
        ## 白字蓝底标签
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0),
                fontsize=3, color='white')## fontsize=14； alpha=0.5

    ## 标题
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #              fontsize=14)

    #plt.tight_layout() ### tight_layout会自动调整子图参数,使之填充整个图像区域


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im_file = os.path.join(file_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # {:d}个对象检测耗时{:.3f}秒

    # Visualize detections for each class
    # 可视化每个类的检测
    CONF_THRESH = 0.85  # 置信度，可视化置信度大于0.6的框
    NMS_THRESH = 0.1  # 示非极大值抑制，这个值越小表示要求的红框重叠度越小，0.0表示不允许重叠
    im = im[:, :, (2, 1, 0)]  # 1,2,0绿色；0,1,2蓝色
    fig, ax = plt.subplots(figsize=(12, 12))  # figsize代表像素值
    ax.imshow(im, aspect='equal')

    label_total = []
    score_totatl = []
    bbox_total = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        bbox_score = get_bbox(cls_ind, dets, thresh=CONF_THRESH)  # 每一类的bbox
        if len(bbox_score) != 0:
            cls_id_num = len(bbox_score)
            for i in range(cls_id_num):
                label_total.append(cls_ind)
                score_totatl.append(bbox_score[i][-1])
                bbox_total.append(bbox_score[i][:-4])

        vis_detections(ax, im, cls, dets, thresh=CONF_THRESH)
        plt.axis('off')  ## #不显示坐标尺寸
        plt.draw()

        ##去白边
        height, width, channels = im.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    # print('dddd')
    return label_total, score_totatl, bbox_score


def mulite_demo(sess, net, im, NMS_thred=None):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # {:d}个对象检测耗时{:.3f}秒

    # Visualize detections for each class
    # 可视化每个类的检测
    CONF_THRESH = NMS_thred  # 置信度，可视化置信度大于0.6的框
    NMS_THRESH = 0.3  # 示非极大值抑制，这个值越小表示要求的红框重叠度越小，0.0表示不允许重叠
    label_total = []
    score_total = []
    bbox_total = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        bbox_score = get_bbox(cls_ind, im, dets, thresh=CONF_THRESH)  # 每一类的bbox
        if len(bbox_score) != 0:
            cls_id_num = len(bbox_score)
            for i in range(cls_id_num):
                label_total.append(cls_ind)
                score_total.append(bbox_score[i][-1])
                bbox_total.append(bbox_score[i][:4])

    # print(len(label_total), len(score_total), len(bbox_total))
    label_total = np.array(label_total)
    score_total = np.array(score_total)
    bbox_total = np.array(bbox_total)
    return label_total, score_total, bbox_total


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


def call_Detection(input, thred=0.5):
    print(thred)
    if thred is None:
        thred = 0.5
    image_path = input
    NMS_thred = thred

    basename = 'Detection_result_Thred_' + str(thred) + '_' + osp.basename(image_path)
    dirname = osp.dirname(image_path)

    if not osp.exists(dirname):
        os.mkdir(dirname)

    savedir = osp.join(dirname, 'out_result/')
    if not osp.exists(savedir):
        os.mkdir(savedir)

    savename = osp.join(savedir, basename)

    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset

    tfmodel = os.path.join('./Crop_Detection/trained_models', 'default_0.0001_DataAug2', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print('tfmodel: ', tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.6

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)

    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    paths = []
    paths.append(image_path)
    for im_name in paths:  # 大图
        print('---------------------------------------------------------')
        print('Demo for data/demo/{}'.format(im_name))
        labels_total = []
        scores_total = []
        bboxes_total = []

        import time
        start_time = time.time()
        sub_img, site = PSP.splitimage(image_path, shape=[416, 416], strided=300)

        for image in sub_img:  # 子图

            labels_, scores_, bboxes_ = mulite_demo(sess, net, image, NMS_thred=NMS_thred)  # 单张子图结果
            num_bbox = len(labels_)
            labels_total.append(labels_)
            scores_total.append(scores_)
            bboxes_total.append(bboxes_)

        print('num of  bbo：', len(bboxes_total))

        labels_merge, scores_merge, bboxes_merge = PSP.merge_label(labels_total, scores_total, bboxes_total, site,
                                                                   [500, 500], im.shape)

        result_img, shapes = draw_toolbox.bboxes_draw_on_img(im, labels_merge, scores_merge, bboxes_merge)

        label_id_dict = draw_toolbox.gain_translate_table()
        # labels_merge = labels_merge.tolist()
        result_num = {}
        for i in range(len(labels_merge)):
            temp = labels_merge[i]
            if temp not in result_num.keys():
                result_num[temp] = 1
            else:
                result_num[temp] = result_num[temp] + 1
        class_num = ''
        for key, value in result_num.items():
            if key in label_id_dict.keys():
                class_num = class_num + label_id_dict[key] + "_" + str(value)
        end_time = time.time()

        time_image = end_time - start_time
        jsonname = 'Thred_' + str(thred) +osp.basename(image_path).split('.')[0] + '.json'
        filename = osp.join(savedir, jsonname)

        from save_json import Save_Json
        json_result = Save_Json()
        json_result.save(filename, shapes, im, NMS_thred)
        cv2.imwrite(savename, result_img)

    print('--------------- finish -----------------------')


# if __name__ == '__main__':
#     img = '/home/kfgeo/Models/Crop_Detection/orig_img/test.png'
#     call_Detection(img,thred=0.8)
#     # args = parse_args()
#     # out_path = '/home/kfgeo/Models/Crop_Detection/123/'
#     # # model path
#     # demonet = args.demo_net
#     # dataset = args.dataset
#     # # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
#     # tfmodel = os.path.join('trained_models','default_0.0001_DataAug2', NETS[demonet][0])
#     #
#     # if not os.path.isfile(tfmodel + '.meta'):
#     #     print('tfmodel: ',tfmodel)
#     #     raise IOError(('{:s} not found.\nDid you download the proper networks from '
#     #                    'our server and place them properly?').format(tfmodel + '.meta'))
#     #
#     # # set config
#     # tfconfig = tf.ConfigProto(allow_soft_placement=True)
#     # tfconfig.gpu_options.allow_growth = True
#     #
#     # # init session
#     # sess = tf.Session(config=tfconfig)
#     # # load network
#     # if demonet == 'vgg16':
#     #     net = vgg16(batch_size=1)
#     # # elif demonet == 'res101':
#     #     # net = resnetv1(batch_size=1, num_layers=101)
#     # else:
#     #     raise NotImplementedError
#     #
#     # n_classes = len(CLASSES)
#     # # create the structure of the net having a certain shape (which depends on the number of classes)
#     # net.create_architecture(sess, "TEST", n_classes,
#     #                         tag='default', anchor_scales=[8, 16, 32])
#     # saver = tf.train.Saver()
#     # saver.restore(sess, tfmodel)
#     #
#     # print('Loaded network {:s}'.format(tfmodel))
#     #
#     # # im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
#     # #            '001763.jpg', '004545.jpg']
#     # # file_path = '/kuafugeo/zmy/Faster-RCNN-TensorFlow-Python3/data/test'
#     # file_path ='/home/kfgeo/Models/Crop_Detection/orig_img/'
#     # im_names = os.listdir(file_path)
#     #
#     # for im_name in im_names:  # 大图
#     #     print('---------------------------------------------------------')
#     #     print('Demo for data/demo/{}'.format(im_name))
#     #     # print('\n')
#     #     labels_total = []
#     #     scores_total = []
#     #     bboxes_total = []
#     #     im_file = os.path.join(file_path, im_name)
#     #     # im_name = 'DJI_0017.JPG'
#     #     # im_file = '/kuafugeo/zmy/Faster-RCNN-TensorFlow-Python3/data/demo_big/DJI_0017.JPG'
#     #     import time
#     #     start_time = time.time()
#     #     im = cv2.imread(im_file)
#     #     # labels_list, score_list, bbox_list = demo(sess, net, im_name)
#     #     sub_img, site = PSP.splitimage(im, shape=[500, 500], strided=200)
#     #     #i=0
#     #     for image in sub_img:  # 子图
#     #       #  cv2.imwrite('/kuafugeo/zmy/Faster-RCNN-TensorFlow-Python3/data/17/'+str(i)+'.png',image)
#     #      #   i+=1
#     #         labels_, scores_, bboxes_ = mulite_demo(sess, net, image)  # 单张子图结果
#     #         # print('labels_:', labels_)
#     #         # print('scores_: ', scores_)
#     #         # print('bboxes_: ', bboxes_)
#     #         num_bbox = len(labels_)
#     #         #print('label: ',labels_, 'scores_: ',scores_, 'bboxes_: ',bboxes_)
#     #         #print('len(labels_): ',len(labels_), 'len(scores_): ',len(scores_), 'len(bboxes_): ',len(bboxes_))
#     #         # print('num_bbox: ',num_bbox)
#     #         #if num_bbox != 0:
#     #         labels_total.append(labels_)
#     #         scores_total.append(scores_)
#     #         bboxes_total.append(bboxes_)
#     #             # for i in range(num_bbox):
#     #             #     labels_total.append(labels_[i])
#     #             #     scores_total.append(scores_[i])
#     #             #     bboxes_total.append(bboxes_[i])
#     #     print('num of  bbo：',len(bboxes_total))
#     #     # # labels_ =
#     #     labels_merge, scores_merge, bboxes_merge = PSP.merge_label(labels_total, scores_total, bboxes_total, site, [500, 500],im.shape)
#     #     # print('labels_merge: ',labels_merge)
#     #     # print('len(labels_merge): ',len(labels_merge))
#     #     # num = 0
#     #     # for t in range(len(bboxes_total)):
#     #     #     num = num + len(bboxes_total[t])
#     #     # temp_bbox_total = np.zeros(shape=(num, 4))
#     #     #
#     #     # num = 0
#     #     # for t in range(len(bboxes_total)):
#     #     #     for j in range(len(bboxes_total[t])):
#     #     #        temp_bbox_total[num] = bboxes_total[t][j]
#     #     #        num = num + 1
#     #     #
#     #     # labels_merge, scores_merge = [], []
#     #     # for t in range(len(labels_total)):
#     #     #     for j in range(len(labels_total[t])):
#     #     #         labels_merge.append(labels_total[t][j])
#     #     #
#     #     # for t in range(len(scores_total)):
#     #     #     for j in range(len(scores_total[t])):
#     #     #         scores_merge.append(scores_total[t][j])
#     #     #
#     #     # bboxes_merge = temp_bbox_total
#     #     # labels_merge = np.array(labels_merge)
#     #     # scores_merge = np.array(scores_merge)
#     #     # im = cv2.imread('/kuafugeo/zmy/Faster-RCNN-TensorFlow-Python3/data/0.png')
#     #     result_img = draw_toolbox.bboxes_draw_on_img(im, labels_merge, scores_merge, bboxes_merge)
#     #
#     #     label_id_dict = draw_toolbox.gain_translate_table()
#     #     # labels_merge = labels_merge.tolist()
#     #     result_num = {}
#     #     for i in range(len(labels_merge)):
#     #         temp = labels_merge[i]
#     #         if temp not in result_num.keys():
#     #             result_num[temp] = 1
#     #         else:
#     #             result_num[temp] = result_num[temp] + 1
#     #     class_num = ''
#     #     for key, value in result_num.items():
#     #         if key in label_id_dict.keys():
#     #             class_num = class_num + label_id_dict[key] + "_" + str(value)
#     #     end_time = time.time()
#     #
#     #     time_image = end_time - start_time
#     #
#     #     cv2.imwrite(out_path + im_name[:-4] + '_' + class_num +  '_totalnums_' + str(len(labels_merge))
#     #                 + '_'  + '%.2f' % time_image+ '.png', result_img)
#     #     #print(PSP)
#     #     # #portion = os.path.splitext(im_name)
#     #     # new_file_path = '/kuafugeo/zmy/Faster-RCNN-TensorFlow-Python3/data/new_image/'
#     #     # plt.savefig(new_file_path + im_name, dpi=300)
#     #
#     # print('---------------------------------------------------------')
#     #
#     #
#     # #plt.show()