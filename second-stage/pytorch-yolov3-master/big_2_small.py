import cv2
import numpy as np
import os
import glob
import random

from libs.configs import *
from libs import configs as cfg
from libs.utils import str2num, parse_anno_file, save_xml


# bbox_list = [[120, 220, 200, 300],
#              [80, 230, 150, 330],
#              [240, 260, 320, 320],
#              [130, 150, 230, 280],
#              [140, 280, 180, 420],
#              [50, 230, 100, 330],
#              [450, 200, ]]


def big_to_small(org_bboxs_list, cut_image_bbox, image, num):
    """
    大图-> 小图 的 框
    :param org_bbox:
    :param temp_image_bbox:
    :param C:
    :param image_dd:
    :return:
    """
    object_id = 0
    new_location = []  # 7163*4295
    # 切块的子图 在原图的位置
    h_s, w_s = cfg.window_sizes
    h_st, h_ed, w_st, w_ed = cut_image_bbox[0], cut_image_bbox[1], cut_image_bbox[2], cut_image_bbox[3]
    # if num == 19:
    #     print("fffffffff")
    # 原图矩形框坐标
    org_bboxs = np.array(org_bboxs_list)
    x1 = org_bboxs[:, 0].astype(int)
    y1 = org_bboxs[:, 1].astype(int)
    x2 = org_bboxs[:, 2].astype(int)
    y2 = org_bboxs[:, 3].astype(int)
    labels = org_bboxs[:, 4]

    xx1 = np.maximum(w_st, x1)
    yy1 = np.maximum(h_st, y1)
    xx2 = np.minimum(w_ed, x2)
    yy2 = np.minimum(h_ed, y2)

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    iou_area = w * h  # 交叠面积
    old_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    id = np.where(iou_area > 5)
    # print(id)
    if len(id[0]) != 0:
        for i in id[0]:
            iou_th = iou_area[i] / old_area[i]
            assert 0 < iou_th <= 1
            if iou_th >= cfg.threshold:
                cut_image = image[h_st:h_ed, w_st:w_ed]  # 在原图上截取子图
                # cv2.rectangle(cut_image, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 0, 225), 1)
                # cv2.imwrite(os.path.join(out_path, file_name_local_num + '@' + 'out_contours.png'), cut_image)
                x_min_new = max(0, x1[i] - w_st)
                y_min_new = max(0, y1[i] - h_st)
                x_max_new = min(x2[i] - w_st, w_s-1)
                y_max_new = min(y2[i] - h_st, h_s-1)
                label = labels[i]
                if label in cfg.labels_list:
                    new_location.append([x_min_new, y_min_new, x_max_new, y_max_new, label])

    # for temp_bbox in org_bboxs_list:
    #     # 原始图像中标注的矩形框坐标
    #     x_min, y_min, x_max, y_max, label = temp_bbox[0], temp_bbox[1], temp_bbox[2], temp_bbox[3], temp_bbox[4]
    #
    #     x_l = np.maximum(x_min, w_st)
    #     y_l = np.maximum(y_min, h_st)
    #     x_r = np.minimum(x_max, w_ed)
    #     y_r = np.minimum(y_max, h_ed)
    #
    #     w = np.maximum(0, x_r - x_l + 1)
    #     h = np.maximum(0, y_r - y_l + 1)
    #     old_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    #     iou_area = w * h  # 交叠面积
    #     if iou_area > 10:
    #         iou_th = iou_area / old_area
    #         assert 0 < iou_th <= 1
    #         if iou_th >= cfg.threshold:
    #             cut_image = image[h_st:h_ed, w_st:w_ed]  # 在原图上截取子图
    #             # cv2.rectangle(cut_image, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 0, 225), 1)
    #             # cv2.imwrite(os.path.join(out_path, file_name_local_num + '@' + 'out_contours.png'), cut_image)
    #             x_min_new = x_l - w_st
    #             y_min_new = y_l - h_st
    #             x_max_new = x_r - w_st
    #             y_max_new = y_r - h_st
    #             new_location.append([x_min_new, y_min_new, x_max_new, y_max_new, label])
    #         # print('ffff', x_min_new, y_min_new, x_max_new, y_max_new)
    #         # print('存在框在切块内部！')

    return new_location


def sliding_over_windows(anno_info, img_xml_path,xml_name):
    """
    对原图图像进行重叠切块，同时读取原图xml中的所有bbox， 再将子图及其对应的xml保存到对应的子文件夹
    :param in_file: 图像路径文件夹
    : image_path: 图像名字
    :out_file: 保存文件夹
    :return:
    """
    image_name = xml_name+'.png'
    image_path = os.path.join(img_xml_path, "tk_pngs", image_name)
    print("image_path: {}".format(image_path))
    image = cv2.imread(image_path)  # 读原图

    all_polygon_list = anno_info["polygon"]

    # 补polygon操作
    for i in range(len(all_polygon_list)):
        seg_list = all_polygon_list[i]["points"]
        seg_list = str2num(seg_list)
        seg_shape = np.array(seg_list).reshape(-1, 1, 2).astype(np.int32)
        tc_ = random.choice(cfg.tudi_list)
        image = cv2.drawContours(image, [seg_shape], -1, tc_, -1)  # (0, 255, 255)

    # box
    all_bbox_list = anno_info["box"]
    plot_img = image.copy()
    for i in range(len(all_bbox_list)):
        if all_bbox_list[i][4] in cfg.not_labels_list:
            tc_ = random.choice(cfg.tudi_list)
            cv2.rectangle(image, (all_bbox_list[i][0], all_bbox_list[i][1]), (all_bbox_list[i][2],
                                                                              all_bbox_list[i][3]), tc_, -1)
        cv2.rectangle(plot_img, (all_bbox_list[i][0], all_bbox_list[i][1]), (all_bbox_list[i][2],
                                                                             all_bbox_list[i][3]), (0, 0, 255), 1)
    #cv2.imwrite("fill_img.png", image)
    #cv2.imwrite("plot_img.png", plot_img)
    image_height = image.shape[0]  # 3648 图像高
    image_width = image.shape[1]  # 4864 图像宽
    num = 0
    h_size, w_size = window_sizes[0], window_sizes[1]
    h_step, w_step = window_steps[0], window_steps[1]
    for y in range(0, image_height, h_step):  # 先 高固定
        for x in range(0, image_width, w_step):  # 然后 宽水平滑动
            # 创建为0的背景图像且其大小为h_size*w_size*3
            img = np.zeros((h_size, w_size, 3), np.uint8)
            window = image[y:y + h_size, x:x + w_size]
            h, w, _ = window.shape
            if h < h_size or w < w_size:
                img[0:h, 0: w] = window
            else:
                img = window

            h_st = y  # 高开始
            w_st = x  # 宽开始
            if h < h_size:
                h_end = image_height  # 超边界
            else:
                h_end = y + h_size
            if w < w_size:
                w_end = image_width
            else:
                w_end = x + w_size

            cut_bbox = [h_st, h_end, w_st, w_end]  # 子图在原图的坐标位置
            filename_localtion =image_name[:-4] + '@' + str(num) + '_' + str(h_st) + '_' + str(h_end) + '_' + str(
                w_st) + '_' + str(w_end)

            image_cpoy = image.copy()
            # # 映射图像块,得到字图对应的xml坐标
            # if num == 19:
            #     print('测试')
            sub_new_location = big_to_small(all_bbox_list, cut_bbox, image_cpoy, num)
            if len(sub_new_location) != 0:
                relative_image_path = filename_localtion + ".png"

                out_save_img_path = os.path.join(img_xml_path, cfg.sub_imgs_file, relative_image_path)

                cv2.imwrite(out_save_img_path, img)  # 保存子图到指定目录

                # plot_img = img.copy()
                # for i in range(len(sub_new_location)):
                #     cv2.rectangle(plot_img, (sub_new_location[i][0], sub_new_location[i][1]),
                #                   (sub_new_location[i][2], sub_new_location[i][3]), (255, 0, 255), 1)
                # cv2.imwrite(os.path.join(out_save_img_path[:-4] + ".png"), plot_img)  # 保存子图到指定目录

                print('该子图{}区域包含目标框: {}'.format(num, len(sub_new_location)))
                out_xml_file = os.path.join(img_xml_path, cfg.sub_xmls_file)
                imgFolderName = out_xml_file.split('/')[-1]
                image_shape = img.shape
                save_xml(imgFolderName, relative_image_path, os.path.join(out_xml_file, relative_image_path),
                         image_shape, sub_new_location, out_xml_file)
            num = num + 1


def bigimg2smallxml(image_xml_path):
    """
    遍历图像文件夹， 图像与xml在同一个文件夹中
    :param images_path:
    :return:
    """
    list_xml_path = glob.glob(os.path.join(image_xml_path, cfg.tk_xmls_file, "*.xml"))  # 列出文件夹下所有的目录与文件

    for temp_xml_path in list_xml_path:
        anno = parse_anno_file(temp_xml_path)
        xml_name=temp_xml_path.split('tk_xmls')[-1][1:-4]
        for i in range(len(anno)):
            img_info = anno[i]
            sliding_over_windows(img_info, image_xml_path,xml_name)


if __name__ == '__main__':
    base_path = r'E:\pytorch-yolov3\yolov3-master\cvatdataset\retrain3'
    #out_image_xml_path = r'./cvatdataset/miaozipo/erqi/dk_1/sub_imgs'
    #save_xml_path = out_image_xml_path
    bigimg2smallxml(base_path)
