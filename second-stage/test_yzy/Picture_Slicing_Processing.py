import numpy as np
import cv2
import numpy

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
    print(type(im_data))
    return im_data

def splitimage(img_path, shape, strided):
    # print(img.size[0],img.size[1])
    in_img = readTif(img_path)
    high = in_img.shape[0]
    width = in_img.shape[1]  # 大图的尺寸
    img_sub = []
    site = []

    for h in range(0, high, strided):
        for w in range(0, width, strided):
            img = numpy.zeros([shape[0], shape[1], 3])
            img[0:min(h + shape[0], high) - h, 0:min(w + shape[1], width) - w, :] = in_img[h:min(h + shape[0], high),
                                                                                    w:min(w + shape[1], width), :]
            img_sub.append(img)
            site.append([h, w])
    return img_sub, site


def merge_label(labels_total, scores_total, bboxes_total, site, shape, im_shape):
    l = len(labels_total)
    T = 1;
    labels_merge = np.array([])
    scores_merge = np.array([])
    bboxes_merge = np.array([])

    for i in range(l):
        list = []
        bboxes = bboxes_total[i]
        if isinstance(bboxes, int):
            continue
        for j in range(len(bboxes)):
            # print(j)
            bbox = bboxes[j]
            # p1=(int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            # p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
                continue
            list.append(j)

        if T == 1:
            T = 0
            labels_merge = labels_total[i][list]
            scores_merge = scores_total[i][list]
            bboxes_merge = bboxes_total[i][list]
            # bboxes_merge[:,[0,2]] = bboxes_merge[:,[0,2]]*shape[0]+site[i][0]#)/im_shape[0]
            # bboxes_merge[:,[1,3]] = bboxes_merge[:,[1,3]]*shape[1]+site[i][1]#)/im_shape[1]
            bboxes_merge[:, [0, 2]] = bboxes_merge[:, [0, 2]] + site[i][1]  # )/im_shape[0]
            bboxes_merge[:, [1, 3]] = bboxes_merge[:, [1, 3]] + site[i][0]  # )/im_shape[1]

        else:

            Temp_bboxes = bboxes_total[i][list]
            # for j in list:
            #  Temp_bboxes[:,0] = (Temp_bboxes[:,0]*shape[0]+site[i][0])/im_shape[0]
            #  Temp_bboxes[:,1] = (Temp_bboxes[:,1]*shape[1]+site[i][1])/im_shape[1]
            #  Temp_bboxes[:,2] = (Temp_bboxes[:,2]*shape[0]+site[i][0])/im_shape[0]
            #  Temp_bboxes[:,3] = (Temp_bboxes[:,3]*shape[1]+site[i][1])/im_shape[1]
            Temp_bboxes[:, [0, 2]] = Temp_bboxes[:, [0, 2]] + site[i][1]  # )/im_shape[0]
            Temp_bboxes[:, [1, 3]] = Temp_bboxes[:, [1, 3]] + site[i][0]  # )/im_shape[1]
            # Temp_bboxes[:,2] = (Temp_bboxes[:,2]*shape[0]+site[i][0])/im_shape[0]
            # Temp_bboxes[:,3] = (Temp_bboxes[:,3]*shape[1]+site[i][1])/im_shape[1]

            # Temp_bboxes[:, [0, 2]] = Temp_bboxes[:, [0, 2]] * shape[0] + site[i][0]  # )/im_shape[0]
            # Temp_bboxes[:, [1, 3]] = Temp_bboxes[:, [1, 3]] * shape[1] + site[i][1]  # )/im_shape[1]   备份
            #  print(list)
            #   print(i,len(list))
            #   print(type(labels_total[i][list]))##<class 'numpy.ndarray'>
            labels_merge = np.hstack((labels_merge, labels_total[i][list]))
            scores_merge = np.hstack((scores_merge, scores_total[i][list]))
            bboxes_merge = np.vstack((bboxes_merge, Temp_bboxes))

    keep = py_cpu_nms(bboxes_merge, scores_merge, 0.15)
    labels_merge = labels_merge[keep]
    scores_merge = scores_merge[keep]
    bboxes_merge = bboxes_merge[keep]

    # bboxes_merge[:, [0, 2]] = bboxes_merge[:, [0, 2]] / im_shape[0]
    # bboxes_merge[:, [1, 3]] = bboxes_merge[:, [1, 3]] / im_shape[1]

    print('---------------finish merge---------------')
    return labels_merge, scores_merge, bboxes_merge


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores1=areas*scores
    order = scores1.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    dets = np.array([[0, 1, 4, 5, 0.8], [1, 0, 5, 4, 0.6], [6, 0, 10, 5, 0.75]])
    thresh = 0.4
    pred = py_cpu_nms(dets, thresh)
    # print(pred)
