# -*- coding:utf-8 -*-
#切后的子图通过cvat核查过的地膜陇距xml来求中心点陇距并生成xml
import numpy as np
import cv2
import time
import os
from xml.dom.minidom import parse
import re
from houchuli import *
import math
from writexml2cvat import *
class Cal():
    def __init__(self):
        pass

    def read_xml(self,cvatxml_path):
        dom = parse(cvatxml_path)
        rootdata = dom.documentElement
        image_list = rootdata.getElementsByTagName('image')
        for image in image_list:
            name = image.getAttribute("name")
            H = int(image.getAttribute("height"))
            W = int(image.getAttribute("width"))
            polyline_list = rootdata.getElementsByTagName('polyline')

        longdui_arry = []
        for polyline in polyline_list:
            long_arry = []
            label = polyline.getAttribute("label")
            points_list = polyline.getAttribute("points")
            points_list = re.split(r'[,;]', points_list)
            points_list = list(float(points_list[i]) for i in range(len(points_list)))
            if (len(points_list) % 2 == 0):
                for idx in range(0, len(points_list), 2):
                    long_arry.append([int(points_list[idx]), int(points_list[idx + 1])])
            longdui_arry.append(long_arry)
        #print('longdui_arry:', longdui_arry)
        return longdui_arry

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
        return im_data,im_geotrans[1]

    def xiangyuan(self, tif_name):
        img, geotrans = self.readTif(tif_name)
        return geotrans

    def Cal_kb(self, data_line1):
        # loc = []  # 坐标
        # for line in data_line1:
        #     x1, y1 = line[0]
        #     loc.append([x1, y1])
        # loc = np.array(loc)  # loc 必须为矩阵形式，且表示[x,y]坐标
        loc = data_line1.reshape(-1, 2)
        output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)

        k = output[1] / output[0]
        k = list(k)[0]
        b = output[3] - k * output[2]
        b = list(b)[0]

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
    def pingxingline(self, h, w, k, hangju):
        A = []
        Ps = []
        if k < 0:
            rrr = (int(w * abs(k))) + h
            # rrr=int(math.sqrt((w*w+h*h)/(k*k)))
            rr = int(math.sqrt(k * k + 1) * hangju)
            for n in range(0, rrr, rr):
                a2 = [w - 1, int((w - 1) * k + n)]
                a1 = [0, n]
                a_ = np.array([a1, a2])
                P = self.line_points(a1, a2)
                Ps.append(P)
                A.append(a_.reshape(-1, 1, 2))
        else:
            rrr = (int(h / k)) + w
            # rrr=int(math.sqrt((w*w+h*h)/(k*k)))
            rr = int(hangju * math.sqrt(k * k + 1) / k)
            for n in range(0, rrr, rr):
                a2 = [1, int(((1 - n) * k) + h)]
                a1 = [n, h]
                a_ = np.array([a1, a2])
                P = self.line_points(a1, a2)
                Ps.append(P)
                A.append(a_.reshape(-1, 1, 2))
        return A, Ps

    def line_points(self, a1, a2):
        startx = a1[0]
        starty = a1[1]
        endx = a2[0]
        endy = a2[1]
        disx = abs(startx - endx)
        disy = abs(starty - endy)
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

    def line_contours(self, contours, Ps):
        # contours：轮廓
        # Ps:切分线的点集
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
                    if (puanduan <= 3) & (puanduan >= -3):
                        pp.append(Ps[i][j])
                        t += 30
                        kn += 1
                        if kk != 0 and kk != k:
                            pp = pp[0:-2]
                        else:
                            continue
                        if kn & 1 == 1:
                            kk = k
                        else:
                            kk = 0
                    else:
                        continue
                t += 1
            if len(pp) != 0:
                pp_c, long = self.distance_c(pp)
                PP.append(pp_c)
                if len(long) != 0:
                    Long.append(long)
                else:
                    Long.append([0])
        print('----------陇距计算完成----------')
        return PP, Long

    def distance_c(self, pp):
        pp_c = []
        long = []
        if len(pp) & 1 == 1:
            for i in range(1, len(pp) - 1, 2):
                x_c = (pp[i][0] + pp[i + 1][0]) / 2
                y_c = (pp[i][1] + pp[i + 1][1]) / 2
                pp_c.append([x_c, y_c])
        else:
            for i in range(0, len(pp) - 1, 2):
                x_c = (pp[i][0] + pp[i + 1][0]) / 2
                y_c = (pp[i][1] + pp[i + 1][1]) / 2
                pp_c.append([x_c, y_c])

        for j in range(0, len(pp_c) - 1):
            x1 = pp_c[j]
            x1 = np.array(x1)
            x2 = pp_c[j + 1]
            x2 = np.array(x2)
            distance = np.sqrt(np.sum(np.square(x1 - x2)))
            long.append(int(distance))
        return pp_c, long

    def error_points(self, PP, Longju, xy):
        NormalPP = []
        NormalLongju = []
        for pp_c, longju in zip(PP, Longju):
            pp_ = []
            longju_ = []
            for i in range(len(longju)):
                if longju[i] <= int(0.6 / xy) or longju[i] >= int(2.4 / xy):  # ( 0.6/xy <= longju <= 2.4/xy)
                    continue
                else:
                    pp_.append([pp_c[i], pp_c[i + 1]])
                    longju_.append(longju[i])
            NormalPP.append(pp_)
            NormalLongju.append(longju_)
        return NormalPP, NormalLongju

    def visual(self, img_name, PP, A):
        image,_=C.readTif(img_name)
        colour = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        n = 0
        for pp in PP:
            for p in pp:
                p1 = (int(p[0][0]), int(p[0][1]))
                p2 = (int(p[1][0]), int(p[1][1]))
                cv2.line(image, p1, p2, colour[n], 8, cv2.LINE_AA)
                if n < 2:
                    n += 1
                else:
                    n = 0
        cv2.drawContours(image, A, -1, (0, 0, 255), 2)
        cv2.imwrite(out_name2, image)

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

        cv2.drawContours(img0, A, -1, (0, 0, 255), 2)
        cv2.imwrite(out_name2, img0)
        print('----------切分线构造完成----------')
        return contours, Ps, H, W, A

if __name__ == '__main__':
    path='E:/pytorch-Unet/Pytorch-UNet-master/test/tujiaoping/yiqi/dk_1/tk_sub_pngs/tk_10'
    for num in range(0,0+1):
        num=1
        TF = False #kongdongtianchong
        hangju= 200 #line dis
        tif_name = path+'/sub_tifs/'+path.split('/')[-1]+'_'+str(num)+'.tif'
        img_name = path + '/sub_dimo_pngs/' + path.split('/')[-1] + '_' + str(num) + '_result.png'
        name = tif_name.split('/')[-1]
        cvatxml_path=path+'/sub_xmls_cvat/yanmiao_longju_yiqi_'+path.split('/')[-5]+'_'+path.split('/')[-3]+'_'+path.split('/')[-1]+'_'+str(num)+'.xml'
        if not os.path.exists(path+ '/sub_xmls_center'):
            os.makedirs(path+ '/sub_xmls_center')
        save_xml_path = path+ '/sub_xmls_center/yanmiao_longju_yiqi_'+path.split('/')[-5]+'_'+path.split('/')[-3]+'_'+path.split('/')[-1]+'_'+str(num)+'.xml'
        out_name =img_name[:-4] + '_1.png'
        out_name2 = img_name[:-4] + '_2.png'

        start = time.time()
        C = Cal()
        xy = C.xiangyuan(tif_name)
        hangju = int(2.0/xy)  # line distance

        #求出切分线
        contours,Ps,H,W,A=C.linedetect(tif_name,img_name, out_name2,hangju)#输出轮廓，切分线的点集，图像高宽

        #求陇距
        PP, Longju = C.line_contours(contours, Ps) #输出交点坐标，陇距

        #求异常
        PP,Longju=C.error_points(PP,Longju,xy) #输出正确陇距点，异常陇距点
        C.visual(tif_name, PP, A)

        end = time.time()

        #xml
        write_xml_cvat(save_xml_path, name, H, W, PP, Longju)
        print('time:', end - start)
    print('_________________')