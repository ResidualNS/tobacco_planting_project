from utils.models import *
from utils.utils import *
from cfg import config as cfgs


class Yolov3Detection():
    def __init__(self):
        self.img_size = cfgs.model_size
        self.conf_thres = cfgs.conf_thres
        self.iou_thres = cfgs.iou_thres
        self.weights = cfgs.weights_path
        self.names = cfgs.name_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.device = torch_utils.select_device()
        self.load_model()

    def load_model(self):
        self.model = Darknet(cfgs.model_cfg, self.img_size)
        if self.weights.endswith('.pt'):
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
            self.model.to(self.device).eval()
        else:
            print("dddddddddddddddddd")

    def process_data(self, image, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False,
                     scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, ratio, (dw, dh)

    def detect(self, save_image, image):
        img = self.process_data(image, new_shape=self.img_size)[0]
        # Convert

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = torch_utils.time_synchronized()
        print("model run time: {}".format(t2 - t1))
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, multi_label=False, classes=None,
                                   agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                ss = det.cpu().numpy()
                boxes_ = ss[:, 0:4]
                conf_ = ss[:, 4]
                cls_ = ss[:, -1].astype(int)
                if save_image:
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
                cv2.imwrite(save_image, image)
                return cls_, conf_, boxes_
            else:
                return 0, 0, 0

    def predict(self, save_img, Source):
        with torch.no_grad():
            labels_, scores_, bboxes_ = self.detect(save_img, Source)
        return labels_, scores_, bboxes_


if __name__ == '__main__':
    param(True, Source='data/samples')
