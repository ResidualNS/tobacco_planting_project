# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np
# def gamma_trans(img):
#     """
#     gamma 校正
#     使用自适应gamma校正
#     :param img: cv2.imread读取的图片数据
#     :return: 返回的gamma校正后的图片数据
#     """
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     mean = np.mean(img_gray)
#     gamma = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
#     return cv2.LUT(img, gamma_table)
#
# path = '1.png'
# img = cv2.imread(path,cv2.IMREAD_COLOR)
# dst=gamma_trans(img)
# cv2.imwrite('3.png', dst)



# numList = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# i = 0
# j= -1
# N=len(numList)
# #如果将该二维列表比作4行3列的表格...
# while i < N-1: #自行定义跳出条件
#
#     #记录当前所在行并判断是否前进到下一行，依据是当前列是否到达该行的结尾
#     i = (i if j != len(numList[i])-1 else i+1)
#
#     #记录当前所在列并判断是继续前进还是回到列头，依据同样是是否到达该行的结尾
#     j = (j+1 if j != len(numList[i])-1 else 0)
#     print(i,j)
#
#     #print(numList[i][j])



# from itertools import product
# p1=[2,4,6,7,8,10,11,6,9,16,17,18,19,9]
# p2=[3,6,9,12,15]
# pp = []
# t = 0
# kk = 0
#
# kkk=3
# for i,j in product(range(0,10),range(len(p1))):
#     if j != t:
#         continue
#     for k in range(len(p2)):
#         if p1[j]==p2[k]:
#             k0=k
#             if kkk % 3 == 0:
#                 t += 3
#                 kkk += 1
#                 pp.append([i,j,k])
#                 break
#         else:
#             continue
#     t += 1
# print(pp)

# img = cv2.imread('E:/pytorch-Unet/Pytorch-UNet-master/test/miaozipo/yiqi/dk_4/tk_pngs/i.png')
# H=img.shape[0]
# W=img.shape[1]
# imgl=img.copy()
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
#
# img[dst>0.8*dst.max()] = [0,0,0]
# #cv2.imwrite('E:/pytorch-Unet/Pytorch-UNet-master/test/miaozipo/yiqi/dk_4/tk_pngs/ii.png',img)
# imgs=imgl-img
# #cv2.imwrite('E:/pytorch-Unet/Pytorch-UNet-master/test/miaozipo/yiqi/dk_4/tk_pngs/iii.png',imgs)
# imgs=imgs[:,:,0]
# A=[]
# for i in range(H):
#     for j in range(W):
#         if imgs[i,j] == 255:
#             A.append([i,j])
# print(A)
# print('s')
from xml.dom.minidom import parse
import re
import cv2
import numpy as np
import math
import time
# xmlpath='./yanmiao_longju_yiqi_zhaojiaba_dk_6_tk_2.xml'
# dom = parse(xmlpath)
# rootdata = dom.documentElement
# image_list = rootdata.getElementsByTagName('image')
# for image in image_list:
#     name = image.getAttribute("name")
#     H = int(image.getAttribute("height"))
#     W = int(image.getAttribute("width"))
#     polyline_list = rootdata.getElementsByTagName('polyline')
#
# img = cv2.imread(name)
# longdui_arry=[]
# for polyline in polyline_list:
#     long_arry = []
#     label = polyline.getAttribute("label")
#     points_list = polyline.getAttribute("points")
#     points_list = re.split(r'[,;]', points_list)
#     points_list = list(float(points_list[i]) for i in range(len(points_list)))
#     if (len(points_list) % 2 == 0):
#         for idx in range(0, len(points_list), 2):
#             long_arry.append([points_list[idx], points_list[idx + 1]])
#     longdui_arry.append(long_arry)
# print('longdui_arry:', longdui_arry)

#
# def get_voc_palette(contour_num):
#     colormap = np.zeros((contour_num, 3), dtype=int)
#     ind = np.arange(contour_num, dtype=int)
#     for shift in reversed(range(8)):
#         for channel in range(3):
#             colormap[:, channel] |= ((ind >> channel) & 1) << shift
#         ind >>= 3
#     return colormap  # 形状[256, 3]
#
# colormap = get_voc_palette(len(image_point_pair_info) + 1)
# colormap = colormap[1:]
# for key, point_pair in image_point_pair_info.items():
#     point_pair = np.array(point_pair).astype(np.int32)
#     color_id = int(key.split("_")[1])
#     color_value = colormap[color_id]
#     # color_value = tuple(color_value)
#     for i in range(len(point_pair)):
#         point = point_pair[i]
#         cv2.line(img, (point[0], point[1]), (point[2], point[3]),
#                  (int(color_value[0]), int(color_value[1]), int(color_value[2])), 2)
# import numpy as np
# from pandas import Series, DataFrame
# def threshold_cluster(Data_set,threshold):
#     #统一格式化数据为一维数组
#     stand_array=np.asarray(Data_set).ravel('C')
#     stand_Data=Series(stand_array)
#     index_list,class_k=[],[]
#     while stand_Data.any():
#         if len(stand_Data)==1:
#             index_list.append(list(stand_Data.index))
#             class_k.append(list(stand_Data))
#             stand_Data=stand_Data.drop(stand_Data.index)
#         else:
#             class_data_index=stand_Data.index[0]
#             class_data=stand_Data[class_data_index]
#             stand_Data=stand_Data.drop(class_data_index)
#             if (abs(stand_Data-class_data)<=threshold).any():
#                 args_data=stand_Data[abs(stand_Data-class_data)<=threshold]
#                 stand_Data=stand_Data.drop(args_data.index)
#                 index_list.append([class_data_index]+list(args_data.index))
#                 class_k.append([class_data]+list(args_data))
#             else:
#                 index_list.append([class_data_index])
#                 class_k.append([class_data])
#     return index_list,class_k
#
#
# Data_set = [1,1,1, 7, 7, 6, 6, 6,5,5]
# index_list, class_k = threshold_cluster(Data_set, 2.25)
# print(index_list)

# words = ['the','sun','did','not','shine','it','was','too','wet','to','play','so','we','sat','in','the','house','all','that','cold','cold','wet','day']
# W=[]
# for ele in words:
#     L=len(ele)
#     W.append(L)
# myset = set(W)
# for i in myset:
#   print(f'Count[{i:0>2}]={W.count(i):0>2}')
# import cv2
# from PIL import Image
# import numpy as np
# img1=Image.open('1.png')
# img2=np.array(img1)
# img1.save('2.png')
# img3=cv2.imread('2.png')
# cv2.imwrite('3.png',img3)

# #图像拼接
# from pylab import *
# from numpy import *
# from PIL import Image
# # If you have PCV installed, these imports should work
# from PCV.geometry import homography, warp
# from PCV.localdescriptors import sift
#
# # set paths to data folder
# featname = ['C:\\Users\DELL\Desktop\PCV\jmu\panorama/z0' + str(i + 1) + '.sift' for i in range(5)]
# imname = ['C:\\Users\DELL\Desktop\PCV\jmu\panorama/z0' + str(i + 1) + '.jpg' for i in range(5)]
#
# # extract features and match
# l = {}
# d = {}
# for i in range(5):
#     sift.process_image(imname[i], featname[i])
#     l[i], d[i] = sift.read_features_from_file(featname[i])
#
# matches = {}
# for i in range(4):
#     matches[i] = sift.match(d[i + 1], d[i])
#
# # visualize the matches (Figure 3-11 in the book)
# '''
# for i in range(4):
#     im1 = array(Image.open(imname[i]))
#     im2 = array(Image.open(imname[i+1]))
#     figure()
#     sift.plot_matches(im2,im1,l[i+1],l[i],matches[i],show_below=True)
# '''
# # function to convert the matches to hom. points
# def convert_points(j):
#     ndx = matches[j].nonzero()[0]
#     fp = homography.make_homog(l[j + 1][ndx, :2].T)
#     ndx2 = [int(matches[j][i]) for i in ndx]
#     tp = homography.make_homog(l[j][ndx2, :2].T)
#
#     # switch x and y - TODO this should move elsewhere
#     fp = vstack([fp[1], fp[0], fp[2]])
#     tp = vstack([tp[1], tp[0], tp[2]])
#     return fp, tp
#
#
# # estimate the homographies
# model = homography.RansacModel()
#
# fp, tp = convert_points(1)
# H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
#
# fp, tp = convert_points(0)
# H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
#
# tp, fp = convert_points(2)  # NB: reverse order
# H_32 = homography.H_from_ransac(fp, tp, model)[0]  # im 3 to 2
#
# tp, fp = convert_points(3)  # NB: reverse order
# H_43 = homography.H_from_ransac(fp, tp, model)[0]  # im 4 to 3
#
# # warp the images
# delta = 500  # for padding and translation
#
# im1 = array(Image.open(imname[1]), "uint8")
# im2 = array(Image.open(imname[2]), "uint8")
# im_12 = warp.panorama(H_12, im1, im2, delta, delta)
#
# im1 = array(Image.open(imname[0]), "f")
# im_02 = warp.panorama(dot(H_12, H_01), im1, im_12, delta, delta)
#
# im1 = array(Image.open(imname[3]), "f")
# im_32 = warp.panorama(H_32, im1, im_02, delta, delta)
#
# im1 = array(Image.open(imname[4]), "f")
# im_42 = warp.panorama(dot(H_32, H_43), im1, im_32, delta, 2 * delta)
#
# figure()
# imshow(array(im_42, "uint8"))
# axis('off')
# show()

#import torch
import numpy as np
# np_data = np.arange(6).reshape((2,3))#range()是python的内置函数，其返回值是range可迭代对象,arange()是Numpy库中的函数，其返回值是数组对象
# torch_data = torch.from_numpy(np_data)    #从numpy中获得数据，可以用torch.from_numpy()
# tensor2array = torch_data.numpy()
#
# print("numpy形式:\n",np_data)
# print("tenor形式:\n",torch_data)
# print("tensor转numpy:\n",tensor2array)

from sklearn.metrics import roc_auc_score
y_true = np.array([1,1,0,0,1,1,0])
y_scores = np.array([0.8,0.7,0.5,0.5,0.5,0.5,0.3])
print("y_true is ",y_true)
print("y_scores is ",y_scores)
print("AUC is",roc_auc_score(y_true, y_scores))



