import cv2
import numpy
import os
import glob

def splitimage(img_path,mask_path, shape, strided):
    big_img = cv2.imread(img_path)
    big_mask=cv2.imread(mask_path)
    print(mask_path)
    high = big_img.shape[0]
    width = big_img.shape[1] #大图的尺寸

    if high == shape[0] and width == shape[1]:
        return big_img, big_mask
    else:
        for h in range(0, high, strided):
            for w in range(0, width, strided):
                filename_= img_path.split('\\')[-1][:-4] + '@' + str(h) + '_' + str(w) + '.png'

                img = numpy.zeros([shape[0], shape[1], 3])
                img[0:min(h + shape[0], high) - h, 0:min(w + shape[1], width) - w, :] = big_img[h:min(h + shape[0], high),w:min(w + shape[1], width), :]
                # if numpy.all(img == 0):
                #     continue
                # else:
                cv2.imwrite('./data/imgs/'+filename_,img)

                mask = numpy.zeros([shape[0], shape[1], 3])
                mask[0:min(h + shape[0], high) - h, 0:min(w + shape[1], width) - w, :] = big_mask[h:min(h + shape[0], high),w:min(w + shape[1], width), :]
                # if numpy.all(img == 0):
                #     continue
                # else:
                cv2.imwrite('./data/masks/'+filename_,mask[:,:,0])
    return

shape=500,500
strided=500
data_path='./cvatdataset/train/'
img_path_list=glob.glob(data_path+'imgs/*.png')
for img_path in img_path_list:
    print(img_path)
    mask_path = img_path.replace('imgs','masks')
    splitimage(img_path,mask_path,shape,strided)
    #print('finish one')
print('~~~~~~~finish all~~~~~~~~')
