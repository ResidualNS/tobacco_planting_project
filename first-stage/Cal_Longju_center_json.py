# -*- coding:utf-8 -*-
import numpy as np
import cv2
import time
import os
from houchuli import *
import math
from writejson2cvat import *
from numba import jitclass


@jitclass()
class Cal_Longju():
    def __init__(self):
        pass

    def readTif(self, fileName):
        import gdal
        dataset = gdal.Open(fileName)
        if dataset == None:
            print(fileName + "文件无法打开")
            return
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_bands = dataset.RasterCount  # 波段数
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息
        im_blueBand = im_data[2, 0:im_height, 0:im_width]  # 获取蓝波段
        im_greenBand = im_data[1, 0:im_height, 0:im_width]  # 获取绿波段
        im_redBand = im_data[0, 0:im_height, 0:im_width]  # 获取红波段
        # im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
        im_data = cv2.merge([im_blueBand, im_greenBand, im_redBand])
        # print(type(im_data))
        return im_data, im_geotrans[1],im_height,im_width

    def Cal_kb(self, data_line1):
        loc = data_line1.reshape(-1, 2)
        output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)

        k = output[1] / output[0]
        k = list(k)[0]
        b = output[3] - k * output[2]
        b = list(b)[0]

        return output, k, b

    def pingxingline(self, h, w, k, hangju):
        trend = [] #切分线方向可视化
        Ps = [] #点集
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
                trend.append(a_.reshape(-1, 1, 2))
        else:
            rrr = (int(h / k)) + w
            rr = int(hangju * math.sqrt(k * k + 1) / k)
            for n in range(0, rrr, rr):
                a2 = [1, int(((1 - n) * k) + h)]
                a1 = [n, h]
                a_ = np.array([a1, a2])
                P = self.line_points(a1, a2)
                Ps.append(P)
                trend.append(a_.reshape(-1, 1, 2))
        return trend, Ps

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

    def distance_c(self, pp):
        center_point = []
        for i in range(0, len(pp) - 1, 2):
            x_c = (pp[i][0] + pp[i + 1][0]) / 2
            y_c = (pp[i][1] + pp[i + 1][1]) / 2
            center_point.append([x_c, y_c])

        center_point_new=[]
        long = []
        for j in range(0, len(center_point) - 1):
            x1 = center_point[j]
            x2 = center_point[j + 1]
            center_point_new.append([x1, x2])
            x1 = np.array(x1)
            x2 = np.array(x2)
            distance = np.sqrt(np.sum(np.square(x1 - x2)))
            long.append(int(distance))
        return center_point_new, long

    def contoursdetect(self, mask_name):
        """
        # png_name: 原始图像
        # img_name: mask二值图像
        """
        dst = cv2.imread(mask_name)
        dst = dst[:, :, 0]
        ret, binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 读取地膜mask图像，找到所有轮廓

        I = []
        contours_new = []  # 去噪后的陇的集合
        for i in contours:
            if i.size > 333:   #小于2株烟草的短陇直接舍去
                I.append(i.size)
                contours_new.append(i)
        # 计算参与计算的陇的阈值：
        I=sorted(I)
        I=I[1:-2] #排除极大值和极小值对平均陇长度计算的影响
        I_th=0.5*np.mean(I) #比0.5倍平均长度的陇还要小的陇不参与斜率的计算
        K_cal = []
        contours_warn=[]
        for j in contours_new:
            if j.size > I_th:
                line_j, k_j, b_j = self.Cal_kb(j)  # 求每个陇的直线方程
                K_cal.append(k_j)
                contours_warn.append(j)

        print('----------轮廓确定完成----------')
        return contours_new,contours_warn, K_cal

    def linedetect(self, img, K_cal, out_name, hangju):
        """
        # out_name:可视化
        # hangju:自拟定的切分线间隔
        """
        img_H = img.shape[0]
        img_W = img.shape[1]
        K_mean = np.mean(K_cal)
        K_long = -1 / K_mean
        trend, Points = self.pingxingline(img_H, img_W, K_long, hangju)  # 给定斜率和间距画出垂线方向的切分线

        #cv2.drawContours(img, trend, -1, (0, 0, 255), 2)
        #cv2.imwrite(out_name, img)
        print('----------切分线构造完成----------')
        return Points,trend

    def line_contours(self, contours, Points):
        """
        contours：轮廓
        Points:切分线的点集
        """
        Center_Points = []
        Longju = []
        line_contour_points_dict = dict()
        for i in range(len(Points)):
            p_point = []
            t = 0
            kk = 0
            kn = 0
            point_distance_dict = dict()
            for j in range(len(Points[i])):
                if j != t:
                    continue
                for k in range(kk, len(contours)):
                    puanduan = cv2.pointPolygonTest(contours[k], Points[i][j], True)
                    if (puanduan <= 2) & (puanduan >= -2):
                        #cv2.drawContours(zero_mask, [contours[k]], -1, (128), 2)
                        if k not in point_distance_dict.keys():
                            point_distance_dict[k] = [[Points[i][j][0], Points[i][j][1]]]
                        else:
                            point_distance_dict[k].append([Points[i][j][0], Points[i][j][1]])
                        #cv2.circle(zero_mask, (Ps[i][j][0], Ps[i][j][1]), 1, (255), 2)
                        p_point.append(Points[i][j])
                        t += 26
                        kn += 1
                        if kk != 0 and kk != k:
                            p_point.remove(Points[i][j])
                        else:
                            continue
                        if kn & 1 == 1:
                            kk = k
                        else:
                            kk = 0
                    else:
                        continue
                t += 1
            line_contour_points_dict[i] = point_distance_dict
            for key,value in point_distance_dict.items():
                num = len(value)
                #print("line_id: ", i, "contour_id: ", key, "contour_point_num: ", num)
                if num == 1:
                    p_point.remove(tuple(value[0]))
                elif num == 3:
                    p_point.remove(tuple(value[1]))
            if len(p_point) != 0:
                center_point, longju = self.distance_c(p_point)
                if len(longju) != 0:
                    Longju.extend(longju)
                    Center_Points.extend(center_point)
                else:
                    continue
        print('----------陇距计算完成----------')
        return Center_Points, Longju

    def error_points(self, Center_Points, Longju, xiangyuan,longju_gt):
        NormalCenter_Points = []
        NormalLongju = []
        for i in range(len(Longju)):
            if Longju[i] <= int(longju_gt*0.5 / xiangyuan) or Longju[i] >= int(longju_gt*1.5 / xiangyuan):  # ( 0.6/xy <= longju <= 1.8/xy)
                continue
            else:
                NormalCenter_Points.append(Center_Points[i])
                NormalLongju.append(Longju[i])
        return NormalCenter_Points, NormalLongju

    def bangding(self, contours, Center_Points,Longju):
        Center_Points_new = []
        Longju_new = []
        for k in range(len(contours)):
            pp_right = []
            longju=[]
            for i in range(len(Center_Points)):
                puanduan_r = cv2.pointPolygonTest(contours[k], tuple(Center_Points[i][1]), False)
                if puanduan_r == 1:
                    pp_right.append(Center_Points[i])
                    longju.append(Longju[i])
                else:
                    continue
            if len(longju) != 0:
                Longju_new.append(longju)
                Center_Points_new.append(pp_right)
        print('----------绑定完成----------')
        return Center_Points_new, Longju_new

    def visual(self, image, out_name2, PP, contours):
        colour = [(0, 0, 255), (0, 204, 153), (0, 255, 255), (0, 255, 0), (128, 128, 0), (255, 0, 0), (128, 0, 128)]
        n = 0
        for pp in PP:
            for p in pp:
                p1 = (int(p[0][0]), int(p[0][1]))
                p2 = (int(p[1][0]), int(p[1][1]))
                cv2.line(image, p1, p2, colour[n], 8, cv2.LINE_AA)
                if n < 6:
                    n += 1
                else:
                    n = 0
        cv2.drawContours(image, contours, -1, (153, 51, 51), 2)
        cv2.imwrite(out_name2, image)
        print('----------可视化完成----------')

    def Warning(self, K, contours, xiangyuan, out_name1, image):
        # K
        K=np.array(K)
        K=np.arctan(K) / np.pi * 180
        K_abs=np.abs(K)
        K_mean=np.mean(K_abs[1:-2])
        if (K > 0).all or (K < 0).all:
            K=K_abs

        err_K = []
        # long
        err_long = []
        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            x, y = rect[0]
            long_w, long_h = rect[1]
            if long_h >= (2.0 / xiangyuan) and long_w >= (2.0 / xiangyuan):  # 由于正常陇的宽度大约为0.8米，地膜间距约为0.6米，所以连陇阈值取2.0米
                err_long.append(i)
                cv2.circle(image, (int(x), int(y)), 20, (147, 20, 255), thickness=-1)
                cv2.putText(image, 'Error_long !', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 5, (147, 20, 255), 5)
            else:
                if abs(K[i]-K_mean) > 15 and abs(K[i]-K_mean) < 30: #斜率阈值0.45 误差为24°，陇距误差0.1米，误差8%
                    err_K.append(i)
                    cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), thickness=-1)
                    cv2.putText(image,'Error_K!', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
                elif abs(K[i]-K_mean) > 30:
                    err_K.append(i)
                    cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), thickness=-1)
                    cv2.putText(image, 'cut!', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
        if len(err_long) != 0 or len(err_K) != 0:
            cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
            cv2.imwrite(out_name1, image)
        return err_K, err_long

if __name__ == '__main__':
    path = './test/zhaojiaba/yiqi/dk_4/'
    for num in range(1, 1 + 1):
        tif_name = path + 'tk_tifs/tk_' + str(num) + '.tif' #原始图像 输入接口
        mask_name = path + 'tk_masks/tk_' + str(num) + '_mask.png' #mask图像 输入接口
        if not os.path.exists(path + 'tk_json'):
            os.makedirs(path + 'tk_json')
        save_json_path = path + 'tk_json/tk_' + str(num) + '.json'
        out_name1 = mask_name[:-9] + '_err.png'
        if not os.path.exists(path + 'tk_visual'):
            os.makedirs(path + 'tk_visual')
        out_name2 = path + 'tk_visual/tk_' + str(num) + '.png'

        start = time.time()
        print('----------开始处理：', tif_name)
        C = Cal_Longju()
        img, xiangyuan, H, W = C.readTif(tif_name)
        pix_num=378956970   #图像大小限制参数
        if H*W > pix_num:
            print('big image!')
            continue
        hangju = int(2.0 / xiangyuan)  #间隔距离参数
        longju_gt = 1.2

        contours,contours_warn,K_cal= C.contoursdetect(mask_name)   # 输出轮廓
        #预警
        err_K,err_long=C.Warning(K_cal,contours_warn,xiangyuan,out_name1,img)
        if len(err_long) > 1 or len(err_K) > 0:
            contours={'err_image_path:':str(out_name1),'err_long:':str(err_long),'err_K:':str(err_K)}
            Center_Points = ''
            Longju = ''
            save_json_path=save_json_path[:-5]+'_err.json'
        else:
            # 求出切分线
            qiefen_Points, trend= C.linedetect(img, K_cal, out_name2, hangju)  # 输出切分线的点集
            # 计算陇距
            Center_Points, Longju = C.line_contours(contours, qiefen_Points)  # 输出交点坐标，陇距
            # 剔除异常
            Center_Points, Longju = C.error_points(Center_Points, Longju, xiangyuan,longju_gt)
            # 陇的绑定
            Center_Points, Longju = C.bangding(contours, Center_Points, Longju)
            # 可视化结果
            C.visual(img, out_name2, Center_Points, contours)
            end = time.time()
            print('time:', end - start)
        # 保存json
        output_cvat_json(tif_name, save_json_path, H, W,xiangyuan, contours, Center_Points, Longju)
    print('_________________')
