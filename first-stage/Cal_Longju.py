# -*- coding:utf-8 -*-
#生成地膜陇距xml上cvat核查
import numpy as np
import cv2
import time
import os
from houchuli import *
import math
from writexml2cvat import *

class Cal():
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
        im_blueBand =im_data[2,0:im_height,0:im_width]#获取蓝波段
        im_greenBand =im_data[1,0:im_height,0:im_width]#获取绿波段
        im_redBand =im_data[0,0:im_height,0:im_width]#获取红波段
        #im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
        im_data = cv2.merge([im_blueBand, im_greenBand, im_redBand])
        #print(type(im_data))
        return im_data,im_geotrans[1]

    def xiangyuan(self,tif_name):
        img,geotrans=self.readTif(tif_name)
        return geotrans

    def Cal_kb(self,data_line1):
        # loc = []  # 坐标
        # for line in data_line1:
        #     x1, y1 = line[0]
        #     loc.append([x1, y1])
        # loc = np.array(loc)  # loc 必须为矩阵形式，且表示[x,y]坐标
        loc=data_line1.reshape(-1,2)
        output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)

        k = output[1] / output[0]
        k=list(k)[0]
        b = output[3] - k * output[2]
        b=list(b)[0]

        return output, k, b

    def Cal_k(self,h, w, data_line2):

        loc = []  # 坐标
        for line in data_line2:
            x1, y1 = line[0][0]
            loc.append([x1, y1])
        loc = np.array(loc)

        [vx, vy, x, y] = cv2.fitLine(loc, cv2.DIST_L1, 0, 0.01, 0.01)

        y1 = int((-x * vy / vx) + y)
        y2 = int(((w - x) * vy / vx) + y)
        a1 = (w - 1, y2)
        a2 = (0, y1)
        k = vy / vx

        return a1, a2, k

    def pingxingline(self,h, w, k,hangju):
        A = []
        Ps = []
        if k < 0:
            rrr=(int(w*abs(k)))+h
            #rrr=int(math.sqrt((w*w+h*h)/(k*k)))
            rr=int(math.sqrt(k*k+1)*hangju)
            for n in range(0, rrr,rr):
                a2 = [w - 1, int((w - 1) * k + n)]
                a1 = [0, n]
                a_ = np.array([a1, a2])
                P = self.line_points(a1, a2)
                Ps.append(P)
                A.append(a_.reshape(-1, 1, 2))
        else:
            rrr=(int(h/k)) + w
            #rrr=int(math.sqrt((w*w+h*h)/(k*k)))
            rr=int(hangju*math.sqrt(k*k+1)/k)
            for n in range(0, rrr,rr):
                a2 = [1, int(((1 - n) * k) + h)]
                a1 = [n, h]
                a_ = np.array([a1, a2])
                P = self.line_points(a1, a2)
                Ps.append(P)
                A.append(a_.reshape(-1, 1, 2))
        return A, Ps

    def line_points(self,a1, a2):
        startx = a1[0]
        starty = a1[1]
        endx = a2[0]
        endy = a2[1]
        disx=abs(startx-endx)
        disy=abs(starty-endy)
        Points = []
        if disx >= disy:
            if startx >= endx:
                for x in range(endx, startx):
                    y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                    Points.append((x, y))

            else:
                for x in range(startx, endx):
                    y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                    Points.append((x, y))
        else:
            if starty >= endy:
                for y in range(endy, starty):
                    x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                    Points.append((x, y))

            else:
                for y in range(starty, endy):
                    x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                    Points.append((x, y))

        return Points

    def line_contours(self,contours, Ps):
        #contours：轮廓
        #Ps:切分线的点集
        PP = []
        Long = []
        for i in range(len(Ps)):
            pp = []
            t = 0
            kk = 0
            kn = 0
            for j in range(len(Ps[i])):
                if j != t:
                    continue
                for k in range(kk, len(contours)):
                    puanduan = cv2.pointPolygonTest(contours[k], Ps[i][j], True)
                    if (puanduan <= 2) & (puanduan >= -2):
                        pp.append(Ps[i][j])
                        kn += 1
                        t +=20
                        if kk != 0 and kk != k:
                            pp.remove(Ps[i][j])
                        if kn % 1 == 1:
                            kk = k
                        else:
                            kk = 0
                        break
                    else:
                        continue
                t += 1
            if len(pp) != 0:
                pp_c,long = self.distance(pp)
                PP.append(pp_c)
                if len(long) != 0:
                    Long.append(long)
                else:
                    Long.append([0])
        print('----------陇距计算完成----------')
        return PP, Long

    def distance(self,pp):
        long = []
        for i in range(1, len(pp) - 1, 2):
            x1 = pp[i]
            x1 = np.array(x1)
            x2 = pp[i + 1]
            x2 = np.array(x2)
            distance = np.sqrt(np.sum(np.square(x1 - x2)))
            long.append(int(distance))
        return pp,long


    def error_points(self, PP, Longju):
        Normal = []
        Error = []
        for pp, longju in zip(PP, Longju):
            nor = []
            err = []
            for i in range(len(longju)):
                if longju[i] >= 60 and longju[i] <= 300: #( 0.6/xy <= longju <= 2.4/xy)
                    nor.append(pp[2 * i + 1])
                    nor.append(pp[2 * i + 2])
                elif longju[i] == 0:
                    continue
                elif longju[i] > 300:
                    continue
                else:
                    err.append(pp[2 * i + 1])
                    err.append(pp[2 * i + 2])
            Normal.append(nor)
            Error.append(err)
        return Normal, Error

    def linedetect(self,input_name,img_name,out_name2,hangju):
        #png_name: 原始图像
        #img_name: mask二值图像
        #out_name2:可视化结果
        #hangju:自拟定的切分线间隔
        img0, xiangyuan = self.readTif(input_name)
        dst = cv2.imread(img_name)
        dst=dst[:,:,0]
        H = img0.shape[0]
        W = img0.shape[1]
        ret, binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #读取地膜mask图像，找到所有轮廓
        Line = []
        D = []
        I = []
        for i in contours:
            I.append(i.size)
        I = sorted(I)
        II=np.mean(I)
        #II=np.max(I)
        #II = I[2]
        K=[]
        for i in contours:
            if i.size <II:
                continue
            else:
                line_, k, b = self.Cal_kb(i) #求每个陇的直线方程
                #K.append(k)
                line_ = np.array(line_)
                line_ = line_.reshape(-1, 1, 2)
                Line.append(line_.astype(int))
        [D.append(Line[i][1][0].reshape(-1, 1, 2)) for i in range(len(Line))]
        a1, a2, k_long = self.Cal_k(H, W, D) #拟合出垂线的方程
        # k_mean=np.mean(K)
        # k_long=-1/k_mean
        A, Ps = self.pingxingline(H, W, k_long, hangju) #给定斜率和间距画出垂线方向的切分线

        #cv2.drawContours(img0, contours, -1, (255, 0, 0), 2)
        cv2.line(img0, a1,a2, (0, 0, 255), 3)
        cv2.drawContours(img0, A, -1, (0, 0, 255), 1)
        cv2.imwrite(out_name2, img0)
        print('----------切分线构造完成----------')
        return contours,Ps,H,W

if __name__ == '__main__':
    path='E:/pytorch-Unet/Pytorch-UNet-master/test/tujiaoping/yiqi/dk_1/'
    for num in range(4,4+1):
        TF = False #kongdongtianchong

        png_name = path+'tk_pngs/tk_'+str(num)+'.png'
        img_name = path+'tk_pngs/mask/tk_'+str(num)+'_result.png'
        xml_name=png_name.split('/')[-1]
        if not os.path.exists(path+ 'tk_xml'):
            os.makedirs(path+ 'tk_xml')
        save_xml_path = path+ 'tk_xml/tk_'+str(num)+'.xml'
        out_name = img_name[:-4] + '_1.png'
        out_name2 = img_name[:-4] + '_2.png'

        C = Cal()
        #tif_name = path + 'tk_tifs/tk_' + str(num) + '.tif'
        #xy=C.xiangyuan(tif_name)
        hangju = 200  # line dis(2/xy)

        #求出切分线

        start = time.time()
        contours,Ps,H,W=C.linedetect(png_name,img_name, out_name2,hangju)#输出轮廓，切分线的点集，图像高宽

        #求陇距
        PP, Longju = C.line_contours(contours, Ps) #输出交点坐标，陇距

        #求异常
        N,E=C.error_points(PP,Longju) #输出正确陇距点，异常陇距点
        end = time.time()

        #xml
        write_xml_cvat(save_xml_path, xml_name, H, W, N,E)
        print('time:', end - start)
    print('_________________')