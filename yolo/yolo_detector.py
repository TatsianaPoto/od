import numpy as np
import torch
import cv2


class YoloDetector():
    def __init__(self, trained_model,cuda=True):
  
        print('Loading weights from checkpoint (' + trained_model + ')')

        # to run without internet connection see below
        # <path_to_local_yolo>, source='local'        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=trained_model) 
        if cuda:
            self.model.cuda()
        self.model = self.model.autoshape()



    def detect(self, img, conf=0.5,iou=0.5,size=640):
        self.model.conf = conf
        self.model.iou = iou

        results = self.model(img, size=size)
        return results.xyxy[0] # ONE IMAGE

        