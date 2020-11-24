# -*- coding: utf-8 -*-
import shutil
import os

def objFileName():
    '''生成文件名列表'''
    local_file_name_list = r'./data/test.txt'
    # 指定名单
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        name=i.replace('\n', '')
        name=name.split('/')[-1]
        obj_name_list.append(name)
    return obj_name_list


def copy_img():
    '''复制、重命名、粘贴文件'''
    local_img_name = r'./data/images'
    # 指定要复制的图片路径
    path =r'./data/samples'
    if not os.path.exists(path):
        os.makedirs(path)
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)

if __name__ == '__main__':
    copy_img()
s=1