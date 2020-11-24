# coding=UTF-8
from xml.dom.minidom import parse
import re
import numpy as np
import math
import time
import os
import glob
import xlrd
import xlwt
from xlutils.copy import copy

def nms(dets_pre,dets_gt, thresh):
    """Pure Python NMS baseline."""
    x1_pre = dets_pre[:, 0]
    y1_pre = dets_pre[:, 1]
    x2_pre = dets_pre[:, 2]
    y2_pre = dets_pre[:, 3]
    areas_pre = (x2_pre - x1_pre + 1) * (y2_pre - y1_pre + 1)
    order_pre = areas_pre.argsort()[::-1]

    x1_gt = dets_gt[:, 0]
    y1_gt = dets_gt[:, 1]
    x2_gt = dets_gt[:, 2]
    y2_gt = dets_gt[:, 3]
    areas_gt = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    order_gt = areas_gt.argsort()[::-1]

    keep = []
    areas_0=0 #0~1024
    areas_1=0 #1024~2048
    areas_2=0 #2048~4096
    areas_3=0 #4096~10000
    n=0
    while order_pre.size > n:
        i = order_pre[n]
        xx1 = np.maximum(x1_pre[i], x1_gt[order_gt[0:]])
        yy1 = np.maximum(y1_pre[i], y1_gt[order_gt[0:]])
        xx2 = np.minimum(x2_pre[i], x2_gt[order_gt[0:]])
        yy2 = np.minimum(y2_pre[i], y2_gt[order_gt[0:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas_pre[i] + areas_gt[order_gt[0:]] - inter)
        inds = np.where(ovr >= thresh)[0]
        if len(inds) != 0:
            keep.append(dets_pre[i])
            if 0 < areas_pre[i] < 1024:
                areas_0 += 1
            elif 1024 <= areas_pre[i] < 2048:
                areas_1 += 1
            elif 2048 <= areas_pre[i] < 4096:
                areas_2 += 1
            elif 4096 <= areas_pre[i] < 10000:
                areas_3 += 1
        n += 1
        Areas_=[areas_0,areas_1,areas_2,areas_3]
    return keep,n,Areas_

def contrast(xml_path):
    dom = parse(xml_path)
    rootdata = dom.documentElement
    image_list = rootdata.getElementsByTagName('image')
    for image in image_list:
        name = image.getAttribute("name")
        H = int(image.getAttribute("height"))
        W = int(image.getAttribute("width"))
        box_list = rootdata.getElementsByTagName('box')

    box_num=0
    det=[]
    for box in box_list:
        box_num += 1
        label=box.getAttribute("label")
        xtl = box.getAttribute("xtl")
        ytl = box.getAttribute("ytl")
        xbr = box.getAttribute("xbr")
        ybr = box.getAttribute("ybr")
        if label == 'yanmiao':
            xyxy=[int(float(xtl)),int(float(ytl)),int(float(xbr)),int(float(ybr))]
            #print('xyxy:',xyxy)
            det.append(xyxy)
            dets=np.array(det)
    return dets,box_num

def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")

def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格[追加]写入数据成功！")


def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()

def main(xml_name):
    #xml_path1 = './data/' + xml_name.split('_')[0] + '/erqi/dk_' + xml_name.split('_')[2] + '/tk_pngs/save_result_new/' + xml_name
    xml_path2 = './data/' + xml_name.split('_')[0] + '/erqi/dk_' + xml_name.split('_')[2] + '/tk_pngs/save_result_3_recall/' + xml_name
    xml_path = './pingtaiwenjian/' + xml_name.split('_')[0] + '/dk_' + xml_name.split('_')[2] + '/tk_' + xml_name.split('_')[4] + '.xml'
    #dets1,num1=contrast(xml_path1)
    dets2,num2=contrast(xml_path2)
    dets,num=contrast(xml_path)
    #keep1,n1=nms(dets1,dets,0.5)
    #keep1_num=len(keep1)
    keep2,n2,Areas_2=nms(dets2,dets,0.5)
    keep2_num=len(keep2) #TP
    loujianshu = num - keep2_num
    wujianshu = num2-keep2_num
    #print(Areas_2)

    #keep1_ap = format(keep1_num / num, '.3f')
    keep2_ap = format(keep2_num / num,'.3f')
    loujian_ap=format(loujianshu / num,'.3f')
    wujian_ap=format(wujianshu / num,'.3f')

    print(keep2_ap)
    value=[[xml_name[:-11],num,keep2_num,float(keep2_ap),loujianshu,float(loujian_ap),wujianshu,float(wujian_ap),Areas_2[0],Areas_2[1],Areas_2[2],Areas_2[3]],]
    write_excel_xls_append(book_name_xls, value)

book_name_xls = '烟苗评价指标统计3.xls'
sheet_name_xls = '二期'
value_title = [["地块ID", "gt数","TP数","识别率","漏检数","漏检率","误检数","误检率","box(<1024)","box(<2048)","box(<4096)","box(<10000)"], ]
#write_excel_xls(book_name_xls, sheet_name_xls, value_title)
start_time = time.time()

dk_name='zhaojiaba_dk_7'
pingtai_path='./pingtaiwenjian/'+dk_name[:-5]+'/'+dk_name[-4:]
num=len(glob.glob(pingtai_path+'/*'))
for i in range(1,num+1):
    xml_name=dk_name+'_tk_'+str(i)+'_result.xml'
    main(xml_name)
end_time = time.time()
print("dk_name:",dk_name)
print('time:', end_time - start_time)
print('--------------- finish-----------------------')

