# =============================================================================
import cv2
import matplotlib.cm as mpcm
import matplotlib.pylab as plt
import json
import numpy as np


def gain_translate_table():
    label2name_table = {}
    #for class_name, labels_pair in dataset_common.VOC_LABELS.items():
        #label2name_table[labels_pair[0]] = class_name
    return label2name_table

label2name_table = gain_translate_table()

def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


colors = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(205, 205, 205), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 1
    shapes = []
    for i in range(bboxes.shape[0]):
        # if classes[i] < 1:
        #     continue
        bbox = bboxes[i]
        #color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue

        ret = cv2.rectangle(img, p1, p2, (0, 0, 255), thickness) #画矩形框
        #print('ret')
        s = '%s/%.1f%%' % ('yanmiao', scores[i]*100)  # 标签
        p = []
        p.append(p1)
        p.append(p2)

       #shp = json_form(label=label2name_table[classes[i]], points_list=p, scores=s, shape_type='rectangle')
        #shapes.append(shp)
        #text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])
        #cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)
        #cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), text_thickness, line_type)
        cv2.putText(img, s, (p1[0], p1[1]), cv2.FONT_HERSHEY_COMPLEX, scale, (255,0,0),thickness=text_thickness, lineType=line_type)
        plt.show()

    # out_path = '/home/kfgeo/Models/faster_rcnn/Faster-RCNN/123/'
    # name = 'test.png'
    # filename = out_path + name.split('.png')[0] + '.json'
    # data = dict(
    #     version='3.6',
    #     imgWidth=str(shape[0]),
    #     imgHeigh=str(shape[1]),
    #     imgBand=str(shape[2]),
    #     shapes=shapes,
    #     imgData='null',
    #     imgPath=filename
    # )
    #
    #
    # with open(filename, 'w') as fid:
    #     json.dump(data, fid, ensure_ascii=True, indent=2, cls=MyEncoder)
    print('---------------finish draw---------------')
    return img, shapes

