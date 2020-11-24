import argparse
import logging
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
import time
import glob
import math
import cv2
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

class pre():
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
        im_blueBand =  im_data[2,0:im_height,0:im_width]#获取蓝波段
        im_greenBand = im_data[1,0:im_height,0:im_width]#获取绿波段
        im_redBand =   im_data[0,0:im_height,0:im_width]#获取红波段
        #im_nirBand = im_data[3,0:im_height,0:im_width]#获取近红外波段
        im_data = cv2.merge([im_blueBand, im_greenBand, im_redBand])
        #print(type(im_data))
        return im_data

    def predict_img(self,net,
                    full_img,
                    device,
                    scale_factor=1,
                    out_threshold=0.5):
        net.eval()

        img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)

            if net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img.size[1]),
                    transforms.ToTensor()
                ]
            )

            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask > out_threshold

    def mask_to_image(self,mask):
        return Image.fromarray((mask * 255).astype(np.uint8))

    #定义图像切分函数
    def image_split(self,img_path,out_files, shape):
        big_img =self.readTif(img_path)
        high = big_img.shape[0]
        width = big_img.shape[1] #大图的尺寸
        strided=shape[0] #子图尺寸就是步长500*500

        if high == shape[0] and width == shape[1]:
            return big_img
        else:
            for h in range(0, high, strided):
                for w in range(0, width, strided):
                    filename_= img_path.split('/')[-1][:-4] + '@' + str(h) + '_' + str(w) + '.png'
                    sub_img = np.zeros([shape[0], shape[1], 3],dtype=np.uint8)
                    sub_img[0:min(h + shape[0], high) - h, 0:min(w + shape[1], width) - w, :] = big_img[h:min(h + shape[0], high),w:min(w + shape[1], width), :]
                    sub_path= out_files.replace('output','subimgs')
                    if not os.path.exists(sub_path):
                        os.makedirs(sub_path)
                    cv2.imwrite(sub_path+'/'+filename_,sub_img)

            print('切图完成!')
            return sub_path,high,width,strided


    # 定义图像拼接函数
    def image_compose(self,submasks,in_name, H, W, S):
        H_num=math.ceil(H/S)
        W_num=math.ceil(W/S)
        big_mask =Image.new('RGB',(W_num*S,H_num*S))  # 创建一个新图
        for h in range(0, H_num):
            for w in range(0, W_num):
                from_img_name=submasks+'/'+in_name+'@'+str(h*S)+'_'+str(w*S)+'.png'
                from_img=Image.open(from_img_name)
                big_mask.paste(from_img,(w*S,h*S))
        box=(0,0,W,H)
        big_mask = big_mask.crop(box)
        return big_mask

    def detect(self,sub_path):
        sub_path2 = sub_path.replace('subimgs', 'submasks')
        if not os.path.exists(sub_path2):
            os.makedirs(sub_path2)
        net = UNet(n_channels=3, n_classes=1)

        logging.info("Loading model {}".format(self.get_args().model))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        net.load_state_dict(torch.load(self.get_args().model, map_location=device))

        mask_list= glob.glob(sub_path+'/*.png')
        for i, fn in enumerate(mask_list):
            logging.info("\nPredicting image {} ...".format(fn))

            img = Image.open(fn)

            mask = self.predict_img(net=net,
                               full_img=img,
                               scale_factor=self.get_args().scale,
                               out_threshold=self.get_args().mask_threshold,
                                device=device)

            out_fn =fn.replace('subimgs','submasks')
            result =self.mask_to_image(mask)
            result.save(out_fn)
        print('预测完成！')
        return sub_path2

    def get_args(self):
        parser = argparse.ArgumentParser(description='Predict masks from input images',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--model', '-m', default='./checkpoints/unet_epoch10.pth',
                            metavar='FILE',
                            help="Specify the file in which the model is stored")
        parser.add_argument('--mask-threshold', '-t', type=float,
                            help="Minimum probability value to consider a mask pixel white",
                            default=0.5)
        parser.add_argument('--scale', '-s', type=float,
                            help="Scale factor for the input images",
                            default=0.5)

        return parser.parse_args()

    def kongdongtianchong(self,img,TF=True):
        img = img[:, :, 0]
        if TF:
            se0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            img = cv2.dilate(img, se0)
            img = cv2.erode(img, se0)
            mask = 255 - img

            # 构造Marker
            SE=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            marker=cv2.erode(mask,SE)

            # 形态学重建
            se = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(25, 25))
            while True:
                marker_pre = marker
                dilation = cv2.dilate(marker, kernel=se)
                marker = np.min((dilation, mask), axis=0)
                if (marker_pre == marker).all():
                    break
            dst = 255 - marker
            dst=cv2.erode(dst,cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15)))
            dst=cv2.dilate(dst,cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15)))
            print('孔洞填充完成!')
            return dst
        else:
            return img

if __name__ == '__main__':
    pre=pre()
    path='./test/tujiaoping/yiqi/'
    L=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.png' and os.path.splitext(file)[0][-6:] != 'result':
                l=os.path.join(root, file).replace('\\','/')
                L.append(l)

    for in_files in L:
        in_name=in_files.split('/')[-1][:-4]
        out_files = path+in_files.split('/')[-3]+'/'+in_files.split('/')[-2]+'/output'
        if not os.path.exists(out_files):
            os.makedirs(out_files)

        start = time.time()
        sub_path, H, W ,S=pre.image_split(in_files,out_files,(3000,3000))
        sub_path2=pre.detect(sub_path)
        result=pre.image_compose(sub_path2,in_name,H,W,S)
        #result.save(out_name)
        result=np.array(result)
        print('合并完成！')

        TF = True
        dst = pre.kongdongtianchong(result, TF)
        out_name = out_files + '/' + in_name + '_result.png'
        cv2.imwrite(out_name, dst)
        print('孔洞填充完成！')

        shutil.rmtree(sub_path)
        shutil.rmtree(sub_path2)
        print('删除完成！')
        end=time.time()
        print('%s完成！'%in_files)
        print('time:',end-start)
    print('python predict.py -i image.png -o output.png')