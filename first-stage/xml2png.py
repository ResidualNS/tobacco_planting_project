from xml.dom.minidom import parse
import re
import cv2
import numpy as np
import math
import time
import glob

def readxml(xmlpath):
    dom = parse(xmlpath)
    rootdata = dom.documentElement
    image_list = rootdata.getElementsByTagName('image')
    for image in image_list:
        name = image.getAttribute("name")
        H = int(image.getAttribute("height"))
        W = int(image.getAttribute("width"))
        polygon_list = image.getElementsByTagName('polygon')

        polygon_arr=[]
        for polygon in polygon_list:
            points_arr = []
            label=polygon.getAttribute("label")
            points_list=polygon.getAttribute("points")
            points_list=re.split(r'[,;]', points_list)
            points_list=list(int(float(points_list[i])) for i in range(len(points_list)))
            if (len(points_list) % 2 == 0):
                for idx in range(0, len(points_list), 2):
                    points_arr.append([points_list[idx], points_list[idx + 1]])
            points_arr.append(points_arr[0])
            points_array=np.array(points_arr)
            #print('points_arr:', points_arr)
            #print('points_array:', points_array)
            polygon_arr.append(points_array)
        polygon_array=np.array(polygon_arr)

        img = np.zeros((H, W))
        cv2.fillPoly(img, pts=polygon_array, color=(255,255,255))
        cv2.imwrite('./cvatdataset/train/masks/'+name,img)
        #cv2.imshow(" ", img)
        #cv2.waitKey(0)

path='./cvatdataset/xml/'
xml_path_list=glob.glob(path+'*.xml')
start_time = time.time()
for xml_path in xml_path_list:
    readxml(xml_path)
end_time = time.time()
print('time:',end_time-start_time)