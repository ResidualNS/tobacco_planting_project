#!C:\Program Files\pythonxy\python\python.exe
# -*- coding:gb2312 -*-
from osgeo import ogr, osr, gdal
import os
import numpy as np
import cv2
import math
from lxml import etree
import glob

"""
Understanding OGR Data Type:
Geometry  - wkbPoint,wkbLineString,wkbPolygon,wkbMultiPoint,wkbMultiLineString,wkbMultiPolygon
Attribute - OFTInteger,OFTReal,OFTString,OFTDateTime
"""

class ARCVIEW_SHAPE:
    # ------------------------------
    # read shape file
    # ------------------------------
    def write_xml_cvat(self,xml_path, img_name, height, width, polyline):
        from lxml import etree

        # 1级目录
        annotation = etree.Element("annotation")

        # 2级目录
        etree.SubElement(annotation, "version").text = "1.1"
        meta = etree.SubElement(annotation, "meta")
        # image_name = img_path.split('\\')[-1]
        image = etree.Element("image", {"id": '0', "name": img_name, "width": str(width), "height": str(height)})
        annotation.append(image)

        # 3级目录
        task = etree.SubElement(meta, "task")
        etree.SubElement(meta, "dumped").text = "2020-08-26 08:22:06.258023+00:00"

        # 4级目录
        etree.SubElement(task, "id").text = "130"
        etree.SubElement(task, "name").text = "task_name"
        etree.SubElement(task, "size").text = "1"
        etree.SubElement(task, "mode").text = "annotation"
        etree.SubElement(task, "overlap").text = "0"
        etree.SubElement(task, "bugtracker")
        etree.SubElement(task, "created").text = "2020-05-20 01:38:39.362995+00:00"
        etree.SubElement(task, "updated").text = "2020-05-20 08:00:34.872446+00:00"
        etree.SubElement(task, "start_frame").text = "0"
        etree.SubElement(task, "stop_frame").text = "0"
        etree.SubElement(task, "frame_filter").text = ' '
        etree.SubElement(task, "z_order").text = "False"

        labels = etree.SubElement(task, "labels")
        # ---包含5,6级目录---
        label = etree.SubElement(labels, "label")
        etree.SubElement(label, "name").text = "dimo"

        segments = etree.SubElement(task, "segments")
        # ---包含5,6级目录---
        segment = etree.SubElement(segments, "segment")
        etree.SubElement(segment, "id").text = "112"
        etree.SubElement(segment, "start").text = "0"
        etree.SubElement(segment, "stop").text = "0"
        etree.SubElement(segment, "url").text = "http://10.10.0.120:8080/?id=112"

        owner = etree.SubElement(task, "owner")
        # ---包含5,6级目录---
        etree.SubElement(owner, "username").text = "kefgeo"
        etree.SubElement(owner, "email").text = "kefgeo@kefgeo.com"

        assignee = etree.SubElement(task, "assignee")
        # ---包含5,6级目录---
        etree.SubElement(assignee, "username").text = "kefgeo"
        etree.SubElement(assignee, "email").text = "kefgeo@kefgeo.com"

        # meta结束，开始image
        # 保存box
        for n in polyline:
            # n = ''.join([str(n[i][0]) + ',' + str(n[i][1]) + ';' for i in range(len(n))])[:-1]
            np = str(n[0]) + ',' + str(n[1]) + ';' + str(n[2]) + ',' + str(n[3])
            polyline = etree.Element("polyline", {"points": np, "label": 'normal', "occluded": '0'})
            image.append(polyline)

        # 2种记录点株方式
        # dianzhu_num = etree.Element("image", {"dianzhu_num": str(dianzhu_num)})
        # image.append(dianzhu_num)

        # print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))

        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        xml_string = ET.tostring(annotation)
        dom = minidom.parseString(xml_string)

        with open(xml_path, 'w', encoding='utf-8') as f:
            dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')

        # print(etree.tostring(annotation, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
        # etree.ElementTree(anno_tree).write(save_path, encoding='UTF-8', pretty_print=True)

        print('xml_path:', xml_path)
        print('---------------finish save_xml---------------')

    def read_shp(self, file):
        # open
        ds = ogr.Open(file, False)  # False - read only, True - read/write
        layer = ds.GetLayer(0)
        # layer = ds.GetLayerByName(file[:-4])
        # fields
        lydefn = layer.GetLayerDefn()
        spatialref = layer.GetSpatialRef()  # 空间坐标
        # spatialref.ExportToProj4()
        # spatialref.ExportToWkt()
        geomtype = lydefn.GetGeomType()
        fieldlist = []  # 属性值，即表头信息
        fid_polygon_dict = dict()
        for i in range(lydefn.GetFieldCount()):
            fddefn = lydefn.GetFieldDefn(i)
            fddict = {'name': fddefn.GetName(), 'type': fddefn.GetType(),
                      'width': fddefn.GetWidth(), 'decimal': fddefn.GetPrecision()}
            fieldlist += [fddict]
        # records
        geomlist = []
        reclist = []
        feature = layer.GetNextFeature()  # 得到特征
        while feature is not None:
            geom = feature.GetGeometryRef()  # 坐标信息
            FID_P = feature.GetFID()
            geomlist += [geom.ExportToWkt()]
            polygon_list = geom.ExportToWkt()
            polygon_list = polygon_list[10:-2]
            polygon_list = polygon_list.split(",")
            fid_polygon_dict[FID_P] = polygon_list
            rec = {}
            for fd in fieldlist:
                rec[fd['name']] = feature.GetField(fd['name'])
            reclist += [rec]
            feature = layer.GetNextFeature()
        # close
        ds.Destroy()
        # 分辨率
        return fid_polygon_dict

    def get_mask_img(self, contour, img, pix_value=(1, 1, 1), channel_num=3):
        xmin = contour[:, :, 0].min()
        xmax = contour[:, :, 0].max()

        ymin = contour[:, :, 1].min()
        ymax = contour[:, :, 1].max()
        # cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)
        # cv2.imwrite("img.png", img)
        sub_img = img[ymin:ymax + 1, xmin:xmax + 1]
        # cv2.imwrite("sub_img.png", sub_img)
        shape_ = sub_img.shape
        contour[:, :, 0] = contour[:, :, 0] - xmin  # + 1
        contour[:, :, 1] = contour[:, :, 1] - ymin  # + 1

        # w = xmax - xmin + 1
        # h = ymax - ymin + 1
        h, w = sub_img.shape[:2]
        if channel_num == 3:
            mask = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, pix_value, -1)
        # cv2.imwrite("mask.png", mask*255)
        sub_img = np.multiply(sub_img, mask)
        # cv2.imwrite("final_sub_img.png", sub_img)
        if channel_num == 3:
            area = np.sum(mask[:, :, 0])
            mask = mask[:, :, 0]
        else:
            area = np.sum(mask[:, :])
            mask = mask
        # area = cv2.countNonZero(mask[:, :, 0])
        return sub_img, mask, [xmin, ymin], contour

    def split_polygon_img(self, tif_path, fid_polygon_dict):
        """
        拆分大图到小图
        :param tif_path:
        :param fid_polygon_dict:
        :return:
        """
        tif_name = os.path.basename(tif_path)[:-4]
        base_path = os.path.dirname(os.path.dirname(tif_path))

        sub_base_path = os.path.join(base_path, "tk_sub_pngs")
        if not os.path.exists(sub_base_path):
            os.makedirs(sub_base_path)
        tk_png_path = os.path.join(sub_base_path, tif_name, "sub_tifs")
        if not os.path.exists(tk_png_path):
            os.makedirs(tk_png_path)
        tk_sub_dimo_path = os.path.join(sub_base_path, tif_name, "sub_dimo_pngs")
        if not os.path.exists(tk_sub_dimo_path):
            os.makedirs(tk_sub_dimo_path)
        tk_xmls_cvat_path = os.path.join(sub_base_path, tif_name, "sub_xmls_center")
        if not os.path.exists(tk_xmls_cvat_path):
            os.makedirs(tk_xmls_cvat_path)

        img, img_trans, im_proj = self.get_trans(tif_path)  # 参考坐标系的图像
        img_mask_path = os.path.join(base_path, "tk_pngs","mask", tif_name + "_result.png")

        dimo_mask = cv2.imread(img_mask_path, 0)
        image_copy = img.copy()
        H, W, _ = img.shape
        pix_num = 178956970
        for key, values in fid_polygon_dict.items():
            new_geomlist = ""
            image_xy_list = list()
            for z_i in range(len(values)):
                shp_p = values[z_i].split(" ")
                img_xy = self.geo2imagexy(img_trans, float(shp_p[0]), float(shp_p[1]), [W, H])
                image_xy_list.append(img_xy)
                new_shp_p = self.imagexy2geo(img_trans, img_xy[0], img_xy[1])
                new_shp_p[0] = str(new_shp_p[0])
                new_shp_p[1] = str(new_shp_p[1])
                # s[0] = str(float(s[0]) + dx)
                # s[1] = str(float(s[1]) - dy)
                s = new_shp_p[0] + " " + new_shp_p[1] + ","
                new_geomlist += s
            new_geomlist = new_geomlist[:-1] + "))"
            value = np.array(image_xy_list).reshape(-1, 1, 2)
            value[:, :, 0] = np.clip(value[:, :, 0], 0, W - 1)  # 大于W-1的变为W-1
            value[:, :, 1] = np.clip(value[:, :, 1], 0, H - 1)
            cv2.drawContours(image_copy, [value], -1, (0, 0, 255), 1)
            # cv2.imwrite(os.path.join(base_path, "tk_sub_pngs", tif_name + '_' + str(key) + '_color.png'), image_copy)

            sub_img, sub_img_mask, offset_xy, sub_cnt = self.get_mask_img(value.copy(), img)
            sub_size = sub_img.shape[0] * sub_img.shape[1]
            if sub_size >= pix_num:
                raise Exception("Invalid sub_size!", sub_size, tif_name, key)
            sub_dimo_mask, _, offset_xy1, sub_cnt1 = self.get_mask_img(value.copy(), dimo_mask, pix_value=(1),
                                                                       channel_num=1)
            pix_x_min = value[:, :, 0].min()
            pix_y_min = value[:, :, 1].min()
            new_xy_min = self.imagexy2geo(img_trans, pix_x_min, pix_y_min)
            new_trans = list(img_trans)
            new_trans[0] = new_xy_min[0]
            new_trans[3] = new_xy_min[1]
            new_trans = tuple(new_trans)
            sub_tif_path = os.path.join(tk_png_path, tif_name + '_' + str(key) + '.tif')
            # cv2.imwrite(os.path.join(tk_png_path, tif_name + '_' + str(key) + '.png'),sub_img)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
            self.writeTiff(new_trans, im_proj, sub_img, sub_tif_path)

            # cv2.imwrite(os.path.join(tk_png_path, tif_name + '_' + str(key) + '_mask.png'),
            #             sub_img_mask * 255)
            cv2.imwrite(os.path.join(tk_sub_dimo_path, tif_name + '_' + str(key) + '_result.png'),
                        sub_dimo_mask)
        print(tif_name, "完成写入")

    def get_trans(self, image_path):
        """
       读取tif图获取rgb数据
       :param fileName:
       :return:
        """
        image_data = gdal.Open(image_path)
        if image_data == None:
            print(image_path + "文件无法打开")
            return
        im_width = image_data.RasterXSize  # 栅格矩阵的列数
        im_height = image_data.RasterYSize  # 栅格矩阵的行数
        im_bands = image_data.RasterCount  # 波段数
        band1 = image_data.GetRasterBand(1)
        print(band1)
        print('Band Type=', gdal.GetDataTypeName(band1.DataType))
        im_data = image_data.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        im_geotrans = image_data.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = image_data.GetProjection()  # 获取投影信息
        im_blueBand = im_data[0, 0:im_height, 0:im_width].reshape(im_height, im_width)  # 获取蓝波段
        im_greenBand = im_data[1, 0:im_height, 0:im_width].reshape(im_height, im_width)  # 获取绿波段
        im_redBand = im_data[2, 0:im_height, 0:im_width].reshape(im_height, im_width)  # 获取红波段
        im_nirBand = im_data[3, 0:im_height, 0:im_width].reshape(im_height, im_width)  # 获取近红外波段
        dtype = im_data.dtype
        del image_data
        rgb_image = np.zeros((im_height, im_width, 3), dtype=dtype)
        rgb_image[:, :, 0] = im_redBand[:]
        rgb_image[:, :, 1] = im_greenBand[:]
        rgb_image[:, :, 2] = im_blueBand[:]

        return rgb_image, im_geotrans, im_proj

    def imagexy2geo(self, trans, col_w, row_h):
        """
        根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param trans: 变换矩阵，即图像左上角的坐标，[0] 是x, [3]是y
        :param row:   像素的行号
        :param col: 像素的列号
        :return:  行列号(row, col)对应的投影坐标或地理坐标(x, y)
        """
        px = trans[0] + col_w * trans[1] + row_h * trans[2]  #
        py = trans[3] + col_w * trans[4] + row_h * trans[5]
        return [px, py]

    def geo2imagexy(self, trans, x_col, y_row, img_size):
        """
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        """
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x_col - trans[0], y_row - trans[3]])
        x, y = np.linalg.solve(a, b)
        if x >= 0:
            x = min(int(x), img_size[0])
        else:
            x = 0
        if y >= 0:
            y = min(int(y), img_size[1])
        else:
            y = 0
        xy = [int(x), int(y)]
        return xy  # 使用numpy的linalg.solve进行二元一次方程的求解

    # # ------------------------------
    # # write shape file
    # # ------------------------------
    # def write_shp(self, file, data):
    #     """
    #     坐标系、
    #     :param file:
    #     :param data:
    #     :return:
    #     """
    #     before_trans = self.get_image(before_path)
    #     after_trans = self.get_image(after_path)
    #
    #     gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    #     gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    #     spatialref, geomtype, geomlist, fieldlist, reclist = data
    #     # spatialref坐标系；
    #     # geomtype类型；
    #     # geomlist每个目标的投影坐标,不包含标签；
    #     # fieldlist表头信息 其他属性信息，包括类别，长、宽；
    #     # reclist 每个目标的类别信息，及其他属性信息
    #     # create
    #     driver = ogr.GetDriverByName("ESRI Shapefile")
    #     if os.access(file, os.F_OK):
    #         driver.DeleteDataSource(file)
    #     ds = driver.CreateDataSource(file)
    #     # spatialref = osr.SpatialReference( 'LOCAL_CS["arbitrary"]' )
    #     # spatialref = osr.SpatialReference().ImportFromProj4('+proj=tmerc ...')
    #     layer = ds.CreateLayer(file[:-4], srs=spatialref, geom_type=geomtype)  # 创建新的layer
    #     # print type(layer)
    #     # fields
    #     for fd in fieldlist:  # 创建属性表头
    #         field = ogr.FieldDefn(fd['name'], fd['type'])
    #         if "width" in fd.keys():
    #             field.SetWidth(fd['width'])
    #         if "decimal" in fd.keys():
    #             field.SetPrecision(fd['decimal'])
    #         layer.CreateField(field)
    #     # records
    #     for i in range(len(reclist)):
    #
    #         temp = geomlist[i]
    #
    #         t_type = temp[:7]
    #         new_geomlist = t_type + " (("
    #         zb = temp[10:-2]
    #         zb_list = zb.split(",")
    #         for z_i in range(len(zb_list)):
    #             s = zb_list[z_i].split(" ")
    #             s = self.geo2imagexy(before_trans, float(s[0]), float(s[1]))
    #             s = self.imagexy2geo(after_trans, s[0], s[1])
    #             s[0] = str(s[0])
    #             s[1] = str(s[1])
    #             # s[0] = str(float(s[0]) + dx)
    #             # s[1] = str(float(s[1]) - dy)
    #             s = s[0] + " " + s[1] + ","
    #             new_geomlist += s
    #         new_geomlist = new_geomlist[:-1] + "))"
    #         # geom = ogr.CreateGeometryFromWkt(geomlist[i])
    #         geom = ogr.CreateGeometryFromWkt(new_geomlist)
    #         feat = ogr.Feature(layer.GetLayerDefn())
    #         feat.SetGeometry(geom)
    #         for fd in fieldlist:
    #             # print(fd['name'],reclist[i][fd['name']])
    #             feat.SetField(fd['name'], reclist[i][fd['name']])
    #         layer.CreateFeature(feat)
    #     # close
    #     ds.Destroy()

    def writeTiff(self, im_geotrans, im_proj, im_data, path):
        """
        保存图像为tif图
        :param im_data:
        :param im_width:
        :param im_height:
        :param im_bands:
        :param im_geotrans:
        :param im_proj:
        :param path:
        :return:
        """
        # im_geotrans, im_proj = self.get_trans(post_coor_path)  # 参考坐标系的图像

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_height, im_width, im_bands = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
        # copy_dataset = dataset  # .copy()
        del dataset  # 作为全局变量
        # return dataset
        # return copy_dataset
        print("完成tif图写入")

    def restore_position(self, p_xy, offest_xy):
        """
        将子图的坐标恢复到原图中的坐标
        :param p_xy:
        :param offest_xy:
        :return:
        """
        new_p_xy = []
        for i in range(len(p_xy)):
            p_i = p_xy[i]
            p_x1 = p_xy[i][0] + offest_xy[0]
            p_y1 = p_xy[i][1] + offest_xy[1]
            p_x2 = p_xy[i][2] + offest_xy[0]
            p_y2 = p_xy[i][3] + offest_xy[1]
            new_p_xy.append([p_x1, p_y1, p_x2, p_y2])
        return new_p_xy

    def parse_anno_file(self, cvat_xml):
        """读取标注的有问题的矩形框的xml"""
        root = etree.parse(cvat_xml).getroot()
        anno = []
        for image_tag in root.iter('image'):
            image = {}
            for key, value in image_tag.items():
                image[key] = value
            image['polygon'] = []
            image["box"] = []
            image['polyline']=[]
            for poly_tag in image_tag.iter('polygon'):
                polygon = {'type': 'polygon'}
                for key, value in poly_tag.items():
                    polygon[key] = value
                image['polygon'].append(polygon)
            for box_tag in image_tag.iter('box'):
                box = {'type': 'box'}
                # box = []
                for key, value in box_tag.items():
                    box[key] = value
                # box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                #     box['xtl'], box['ytl'], box['xbr'], box['ybr'])

                new_box = [int(float(box["xtl"])), math.ceil(float(box["ytl"])),
                           int(float(box["xbr"])), math.ceil(float(box["ybr"])),
                           box["label"]]
                image['box'].append(new_box)
            for polyline_tag in image_tag.iter('polyline'):
                polyline={'type':'polyline'}
                for key,value in polyline_tag.items():
                    polyline[key] = value
                new_polyline=[float(polyline['points'].split(';')[0].split(',')[0]),float(polyline['points'].split(';')[0].split(',')[1]),
                              float(polyline['points'].split(';')[1].split(',')[0]),float(polyline['points'].split(';')[1].split(',')[1])]
                image['polyline'].append(new_polyline)

            image['polygon'].sort(key=lambda x: int(x.get('z_order', 0)))
            # image['box'].sort(key=lambda x: int(x.get('z_order', 0)))
            anno.append(image)

        return anno

    def merge_sub_tif(self, tif_path, fid_polygon_dict,cvat_xml):
        """
        合并小图待到大图
        :param tif_path:
        :param fid_polygon_dict:
        :return:
        """
        tif_name = os.path.basename(tif_path)[:-4]
        base_path = os.path.dirname(os.path.dirname(tif_path))

        sub_base_path = os.path.join(base_path, "tk_sub_pngs")
        # if not os.path.exists(sub_base_path):
        #     os.makedirs(sub_base_path)
        tk_png_path = os.path.join(sub_base_path, tif_name, "sub_tifs")
        # if not os.path.exists(tk_png_path):
        #     os.makedirs(tk_png_path)
        tk_sub_dimo_path = os.path.join(sub_base_path, tif_name, "sub_dimo_pngs")
        # if not os.path.exists(tk_sub_dimo_path):
        #     os.makedirs(tk_sub_dimo_path)

        img, img_trans, im_proj = self.get_trans(tif_path)  # 参考坐标系的图像
        # img_mask_path = os.path.join(base_path, "dimo_masks", tif_name + "_result.png")
        # dimo_mask = cv2.imread(img_mask_path, 0)
        image_copy = img.copy()
        H, W, _ = img.shape
        pix_num = 178956970
        ori_box_list = []
        sub_xmls_cvat = "sub_xmls_center"
        for key, values in fid_polygon_dict.items():
            new_geomlist = ""
            image_xy_list = list()
            for z_i in range(len(values)):
                shp_p = values[z_i].split(" ")
                img_xy = self.geo2imagexy(img_trans, float(shp_p[0]), float(shp_p[1]), [W, H])
                image_xy_list.append(img_xy)
                new_shp_p = self.imagexy2geo(img_trans, img_xy[0], img_xy[1])
                new_shp_p[0] = str(new_shp_p[0])
                new_shp_p[1] = str(new_shp_p[1])
                s = new_shp_p[0] + " " + new_shp_p[1] + ","
                new_geomlist += s
            new_geomlist = new_geomlist[:-1] + "))"
            value = np.array(image_xy_list).reshape(-1, 1, 2)
            value[:, :, 0] = np.clip(value[:, :, 0], 0, W - 1)  # 大于W-1的变为W-1
            value[:, :, 1] = np.clip(value[:, :, 1], 0, H - 1)
            cv2.drawContours(image_copy, [value], -1, (0, 0, 255), 1)
            sub_img, sub_img_mask, offset_xy, sub_cnt = self.get_mask_img(value.copy(), img)
            sub_size = sub_img.shape[0] * sub_img.shape[1]
            if sub_size >= pix_num:
                raise Exception("Invalid sub_size!", sub_size)
            save_xml_path = os.path.join(sub_base_path, tif_name, sub_xmls_cvat)
            save_xml_path = os.path.join(save_xml_path, cvat_xml+tif_name + '_' + str(key) + '.xml')
            anno = self.parse_anno_file(save_xml_path)

            for a_i in range(len(anno)):

                points = anno[a_i]["polyline"]
                new_aa = self.restore_position(points, offest_xy=offset_xy)
                color = [0, 0, 255]
                for nwe_a_i in range(len(new_aa)):
                    x = new_aa[nwe_a_i]
                    ori_box_list.append(x)
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    cv2.line(img, c1, c2, color, thickness=5)
                # cv2.imwrite("test_output" + "/ddd.png", img)
        out_ori_img = os.path.join(sub_base_path, tif_name, tif_name + ".png")
        cv2.imwrite(out_ori_img, img)
        out_ori_xml = os.path.join(sub_base_path, tif_name, cvat_xml+tif_name + ".xml")
        self.write_xml_cvat(out_ori_xml, tif_name + ".png", H, W, ori_box_list)


# --------------------------------------
# main function
# --------------------------------------
if __name__ == "__main__":
    img_base_path = 'E:/pytorch-Unet/Pytorch-UNet-master/test/tujiaoping/yiqi/dk_1'
    cvat_xml='yanmiao_longju_yiqi_'+img_base_path.split('/')[-3]+'_'+img_base_path.split('/')[-1]+'_'
    model_shp = ARCVIEW_SHAPE()
    tk_sub_shps = "tk_shps"
    tk_tifs = "tk_tifs"
    shp_path_list = glob.glob(os.path.join(img_base_path, tk_sub_shps, "*.shp"))
    for i in range(len(shp_path_list)):
        shp_path = shp_path_list[i]
        shp_name = os.path.basename(shp_path)[:-4]
        tif_path = os.path.join(img_base_path, tk_tifs, shp_name + ".tif")
        fid_polygon_dict = model_shp.read_shp(shp_path)
        #ss = model_shp.split_polygon_img(tif_path, fid_polygon_dict)  # 拆分
        dd = model_shp.merge_sub_tif(tif_path, fid_polygon_dict,cvat_xml)  # 合并cvat结果
print("dddddd")
