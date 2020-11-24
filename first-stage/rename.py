# -*- coding:utf-8 -*-
import os
import cv2
from shutil import copyfile


def readTif(fileName):
    import gdal
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
    im_proj = dataset.GetProjection()#获取投影信息
    im_blueBand =  im_data[2,0:im_height,0:im_width]#获取蓝波段
    im_greenBand = im_data[1,0:im_height,0:im_width]#获取绿波段
    im_redBand =   im_data[0,0:im_height,0:im_width]#获取红波段
    #im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
    im_data = cv2.merge([im_blueBand, im_greenBand, im_redBand])
    #print(type(im_data))
    return im_data

filepath='./tujiaoping_mask/'
outpath='./test/tujiaoping/yiqi/'
files=os.listdir(filepath)
dklist=[]
for dk_n in files:
    if not os.path.isdir(filepath + '/'+dk_n):   #这里是绝对路径，该句判断目录是否是文件夹
        continue
    else:
        dklist.append(dk_n)
        dk=outpath+dk_n
        if not os.path.exists(dk):
            os.makedirs(dk)
        tk_masks=dk+'/tk_masks'
        if not os.path.exists(tk_masks):
            os.makedirs(tk_masks)
        print(dk)
        tks = os.listdir(filepath+dk_n)
        tklist=[]
        for tk in tks:
            tk_num=tk.split('_')[-2]
            img=filepath+dk_n+'/'+tk
            out=tk_masks+'/tk_'+tk_num+'_mask.png'
            copyfile(img,out)

# from predict import *
# from houchuli import *
#
# #rename
# path='./test/zhaojiaba/yiqi/dk_7'
# in_name='tk_2_1'
# filepath=path+'/tk_tifs/'+in_name
# outpath=filepath.replace('tk_tifs','tk_pngs')
# img=readTif(filepath+'.tif')
# cv2.imwrite(outpath+'.png',img)
#
# #predict
# pre= pre()
# in_files=outpath+'.png'
# out_files = path+ '/' + in_files.split('/')[-2] + '/output2'
# if not os.path.exists(out_files):
#     os.makedirs(out_files)
# sub_path, H, W ,S=pre.image_split(in_files,out_files,(3000,3000))
# sub_path2=pre.detect(sub_path)
#
# result=pre.image_compose(sub_path2,in_name,H,W,S)
# out_name = out_files + '/' + in_name + '_result.png'
# result.save(out_name)
# print('合并完成！')
#
# shutil.rmtree(sub_path)
# shutil.rmtree(sub_path2)
# print('删除完成！')
# #print('%s完成！'%in_files)
#
# #houchuli
# TF=True
# kdtc = kdtc()
# dst=kdtc.kongdongtianchong(out_name,TF)
# mask_files = out_name.replace('output2','mask')
# cv2.imwrite(mask_files, dst)

print('-----------')
