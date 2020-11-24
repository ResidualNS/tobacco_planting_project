# -*- coding:utf-8 -*-
import numpy as np
import cv2
import time
import os
from xml.dom.minidom import parse
import re
from houchuli import *
import math
from writejson2cvat import *
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
                    long_arry.append((int(points_list[idx]), int(points_list[idx + 1])))
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

    def xiangyuan(self,tif_name):
        img,geotrans=self.readTif(tif_name)
        return geotrans

    def Cal_kb(self,data_line1):

        loc = []  # 坐标
        for line in data_line1:
            x1, y1 = line[0]
            loc.append([x1, y1])
        loc = np.array(loc)  # loc 必须为矩阵形式，且表示[x,y]坐标

        output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)

        k = output[1] / output[0]
        b = output[3] - k * output[2]

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
        PP = []
        Long = []
        for k in range(len(contours)):
            pp=[]
            pp_l = []
            pp_r = []
            for i in range(len(Ps)):
                #puanduan_l = cv2.pointPolygonTest(contours[k], Ps[i][0], True)
                puanduan_r = cv2.pointPolygonTest(contours[k], Ps[i][1], False)
                #if (puanduan_l <= 5) & (puanduan_l >= -5):
                    #pp_l.append(Ps[i])
                if puanduan_r == 1:
                    pp_r.append(Ps[i])
                else:
                    continue
            pp=pp_r
            if len(pp) != 0:
                long = self.distance(pp)
                PP.append(pp)
                if len(long) != 0:
                    Long.append(long)
                else:
                    Long.append([])
            else:
                PP.append([])
                Long.append([])
        print('----------陇距计算完成----------')
        return PP, Long

    def distance(self,pp):
        long = []
        for i in range(0, len(pp)):
            x1 = pp[i][0]
            x1 = np.array(x1)
            x2 = pp[i][1]
            x2 = np.array(x2)
            distance = np.sqrt(np.sum(np.square(x1 - x2)))
            long.append(int(distance))
        return long


    def linedetect(self,png_name,img_name,out_name2,hangju):
        img0,_ = self.readTif(png_name)
        dst = cv2.imread(img_name)
        dst=dst[:,:,0]
        H = img0.shape[0]
        W = img0.shape[1]
        ret, binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours,H,W

    def visual(self, png_name, PP):
        image, _ = self.readTif(png_name)
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
        cv2.imwrite(out_name2, image)

if __name__ == '__main__':
    path='E:/pytorch-Unet/Pytorch-UNet-master/test/tujiaoping/yiqi/dk_1/'
    for num in range(10,10+1):
        TF = False #kongdongtianchong
        hangju= 200 #line dis
        png_name = path+'tk_pngs/tk_'+str(num)+'.png'
        #tif_name = path+'tk_tif/tk_'+str(num)+'.tif'
        img_name = path+'tk_pngs/mask/tk_'+str(num)+'_result.png'
        xml_path=path+'tk_xml/tk_'+str(num)+'.xml'
        if not os.path.exists(path+ 'tk_json'):
            os.makedirs(path+ 'tk_json')
        save_json_path = path+'tk_json/tk_'+str(num)+'.json'
        out_name = img_name[:-4] + '_1.png'
        out_name2 = img_name[:-4] + '_2.png'


        #求出切分线
        C = Cal()
        start = time.time()
        contours,H,W=C.linedetect(png_name,img_name, out_name2,hangju)
        longdui=C.read_xml(xml_path)

        #求陇距
        PP, Longju = C.line_contours(contours, longdui)
        end = time.time()
        C.visual(png_name, PP)
        #json
        output_cvat_json(png_name,save_json_path,H, W,contours,PP,Longju)
        print('time:', end - start)
    print('_________________')