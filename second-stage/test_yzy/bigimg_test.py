import Picture_Slicing_Processing as PSP
from utils import draw_toolbox
from utils.detect import *
import cv2
import time
import os
from utils.writexml2cvat import write_xml_cvat

model = Yolov3Detection()

class big_detect():
    def __init__(self):
        pass

    def big_detect(self, im_path, save_img_path, save_xml_path):
        labels_total = []
        scores_total = []
        bboxes_total = []
        im = cv2.imread(im_path)
        H = im.shape[0]
        W = im.shape[1]
        sub_img, site = PSP.splitimage(im_path, shape=[416, 416], strided=300)
        num = 0
        for image in sub_img:  # 子图
            image_name = 'save_sub_img/' + str(num) + '.png'
            labels_, scores_, bboxes_ = model.predict(image_name, Source=image)  # 单张子图结果
            labels_total.append(labels_)
            scores_total.append(scores_)
            bboxes_total.append(bboxes_)
            num += 1
        print('num of  bbo：', len(bboxes_total))

        labels_merge, scores_merge, bboxes_merge = PSP.merge_label(labels_total, scores_total, bboxes_total, site, [416, 416], im.shape)

        result_img, shapes = draw_toolbox.bboxes_draw_on_img(im, labels_merge, scores_merge, bboxes_merge)

        label_id_dict = draw_toolbox.gain_translate_table()

        result_num = {}
        for i in range(len(labels_merge)):
            temp = labels_merge[i]
            if temp not in result_num.keys():
                result_num[temp] = 1
            else:
                result_num[temp] = result_num[temp] + 1
        class_num = ''
        for key, value in result_num.items():
            if key in label_id_dict.keys():
                class_num = class_num + label_id_dict[key] + "_" + str(value)

        cv2.imwrite(save_img_path, result_img)
        img_name = os.path.basename(im_path)[:-4]
        write_xml_cvat(save_xml_path, img_name, H, W, bboxes_merge)
        print('--------------- finish detect -----------------------')

if __name__ == '__main__':
    file_path = r'./data/zhaojiaba/erqi/dk_7/tk_pngs'
    num = len(glob.glob(file_path + '/*.png'))
    for i in range(1,num+1):
        png_name='tk_'+str(i)+'.png'
        im_path = os.path.join(file_path, png_name)
        image_name = os.path.basename(im_path)[:-4]
        save_resrlt=os.path.join(file_path, "save_result_3_recall")
        if not os.path.exists(save_resrlt):
            os.mkdir(save_resrlt)
        save_name=file_path.split('/')[-4]+'_'+file_path.split('/')[-2]+'_'+image_name
        save_img_path = os.path.join(save_resrlt, save_name + '_result.png')
        xml_path = os.path.join(save_resrlt, save_name + '_result.xml')

        start_time = time.time()
        bd = big_detect()
        bd.big_detect(im_path, save_img_path, xml_path)
        end_time = time.time()
        print('time:', end_time - start_time)
    print('--------------- finish-----------------------')
