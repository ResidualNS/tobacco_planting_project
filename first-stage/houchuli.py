# -*- coding:utf-8 -*-
import numpy as np
import cv2
import time
import os

class kdtc():
    def __init__(self):
        pass

    def readTif(self,fileName):
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

    def kongdongtianchong(self,img_name,TF=True):
        img = self.readTif(img_name)
        img = img[:, :, 0]
        if TF:
            se0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            img = cv2.dilate(img, se0)
            img = cv2.erode(img,se0)
            mask = 255 - img

            # 构造Marker
            SE=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            marker=cv2.erode(mask,SE)

            # 形态学重建
            se = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(25, 25))
            while True:
                marker_pre = marker
                dilation = cv2.dilate(marker, kernel=se)
                marker = np.min((dilation, mask), axis=0)
                if (marker_pre == marker).all():
                    break
            dst = 255 - marker
            dst=cv2.erode(dst,cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15)))
            dst=cv2.dilate(dst,cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15)))
            print('孔洞填充完成!')
            return dst
        else:
            return img

if __name__ == '__main__':
    #img_name = 'E:/pytorch-Unet/Pytorch-UNet-master/test/zhaojiaba/yiqi/dk_2/tk_pngs/output/tk_27_result.png'
    #out_name = img_name[:-4] + '_1.png'
    kdtc = kdtc()
    for num in range(1,3):
        path='./test/tujiaoping/yiqi/dk_'+str(num)+'/tk_pngs/output2/'
        L=[]
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == '.png' and os.path.splitext(file)[0][-6:] == 'result':
                    l=os.path.join(root, file).replace('\\','/')
                    L.append(l)

        for in_files in L:
            in_name=in_files.split('/')[-1][:-4]
            out_files = path.replace('output2','mask')
            if not os.path.exists(out_files):
                os.makedirs(out_files)

            TF=True
            start = time.time()

            dst=kdtc.kongdongtianchong(in_files,TF)
            cv2.imwrite(out_files+in_name+'.png', dst)
            end = time.time()
            print('time:', end - start)
        print(path)
    print('---------------')

