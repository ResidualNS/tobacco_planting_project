# -*- coding:utf-8 -*-
import json
import numpy as np
import os
import xlrd
import xlwt
from xlutils.copy import copy

def read_json(json_path):
    jp = open(json_path)
    rootdata = json.load(jp)
    name = rootdata['image_name']

    height = rootdata['image_height']
    width = rootdata['image_width']
    if 'image_xiangyuan' in rootdata:
        xiangyuan = rootdata['image_xiangyuan']
    # else:
    #     xiangyuan = 0.007689999999997638
    point_pair_data = rootdata['point_pair']

    value_list=[]
    for key,value in point_pair_data.items():
        value_list.extend(value)
    value_arr=np.array(value_list,dtype=object)
    value_arr=value_arr[:,-1]
    return name, height, width, xiangyuan, value_arr

def cal_acc(xiangyuan, value_arr):
    longju_gt = 1.2 / xiangyuan
    longju_03 = 0.24 / xiangyuan  #%20
    longju_02 = 0.12 / xiangyuan #%10
    longju_01 = 0.06 / xiangyuan #%5
    value_num=value_arr.size
    value_mean=round(np.mean(value_arr)*xiangyuan, 3)
    value_acc=round(1-(abs(value_mean-1.2))/1.2, 3)
    num_03 = np.count_nonzero(abs(value_arr - longju_gt) < longju_03)
    acc_03= round(num_03/value_num,3)
    num_02 = np.count_nonzero(abs(value_arr - longju_gt) < longju_02)
    acc_02 = round(num_02/value_num,3)
    num_01 = np.count_nonzero(abs(value_arr - longju_gt) < longju_01)
    acc_01 = round(num_01/value_num,3)

    return value_num, value_mean, value_acc, acc_01, acc_02, acc_03

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

if __name__ == '__main__':
    path = './test/miaozipo/yiqi/dk_6'
    for num in range(4,4+1):
        json_path =path+'/tk_json/tk_'+str(num)+'.json'
        name, height, width, xiangyuan, value_arr=read_json(json_path)
        value_num, value_mean, value_acc, acc_01, acc_02, acc_03 = cal_acc(xiangyuan, value_arr)
        xls_path = '烟苗一期评价指标统计.xls'
        sheet_name = '陇距'
        sheet_title = [["区域ID", "地块ID", "田块ID", "陇距数", "陇距均值", "总准确率", "±0.06米范围准确率", "±0.12米范围准确率", "±0.24米范围准确率"], ]
        if not os.path.exists(xls_path):
            write_excel_xls(xls_path, sheet_name, sheet_title)
        value_= [[path.split('/')[-3], path.split('/')[-1], name[:-4], value_num, value_mean, value_acc, acc_01, acc_02, acc_03],]
        write_excel_xls_append(xls_path, value_)
        print("json_name:",json_path)
print('--------------- finish-----------------------')

