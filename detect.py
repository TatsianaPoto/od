
import numpy as np
import argparse
import cv2
import re
from pathlib import Path


from craft import CraftDetector
from yolo import YoloDetector
from classifier import Classifier

from general_utils import four_point_transform,crop,pad

# Not working yet
# from experiments.cc import get_clustered_boxes
# from experiments.segmentation import segment_chars

yolo = YoloDetector('weights/yolo_containers.pt')

params_chars = {'text_threshold':0, 'link_threshold':1, 'low_text':0.48}
params_boxes = {'text_threshold':0, 'link_threshold':0.4, 'low_text':0.4}
craft = CraftDetector('weights/craft_mlt_25k.pth',refine=False)

resnet = Classifier('weights/resnet_symbols.pt')

# Detects container number from whole image
def detect(image,return_strict=False):
    yolo_result = yolo.detect(image,conf=0.1,iou=0.4)

    result = []
    yolo_pad = 25
    for r in yolo_result:
        score = float(r[4])
        x_top,y_top,x_bottom,y_bottom = int(r[0]),int(r[1]),int(r[2]),int(r[3])

        y_top,y_bottom,x_top,x_bottom = pad(y_top,y_bottom,x_top,x_bottom,yolo_pad)

        cropped_image = crop(image,y_top,y_bottom,x_top,x_bottom,0)
        # cv2.imshow('t',cropped_image)
        # cv2.waitKey()

        text = get_text(cropped_image)
        if is_valid(text,strict=return_strict):
            result.append( ([x_top,y_top,x_bottom,y_bottom],score,text) )        

    return result

#Detects container number from small region detected by yolo
def get_text(image):
    # get boxes with text
    bboxes, polys, mask = craft.detect(image,**params_boxes) 
    if len(bboxes) > 0:
        horizontal = is_horizontal(bboxes)
        bboxes = sort_boxes(bboxes,horizontal)
    
    #detect characters in each textbox
    images_with_boxes = []
    for i, box in enumerate(bboxes):
        cropped_image = four_point_transform(image,box,pad_v=2)
        # cv2.imshow('t',cropped_image)
        # cv2.waitKey()

        char_boxes, _, _ = craft.detect(cropped_image,**params_chars)

        if len(char_boxes) > 0:
            char_boxes = sorted(char_boxes, key=lambda x: x[0][1-horizontal])
        else:
            char_boxes = [np.array([[0,0],[cropped_image.shape[1],0],[cropped_image.shape[1],cropped_image.shape[0]],[0,cropped_image.shape[0]]])]
        
        images_with_boxes.append( (cropped_image, char_boxes))

    images_with_boxes = limit_boxes(images_with_boxes)

    # classify each letter in each box
    text = ''
    for i, (cropped_image,char_boxes) in enumerate(images_with_boxes):
        # cv2.imshow('t',cropped_image)
        # cv2.waitKey()
        whitelist = get_whitelist(i,len(images_with_boxes))

        for char_box in char_boxes:             
            text += recognize(cropped_image,char_box,whitelist)
    return text

MIN_PAD = 0
MAX_PAD = 1
#Recognize one symbol from small image
def recognize(image,box,whitelist,min_pad=MIN_PAD, max_pad=MAX_PAD):
    (tl, tr, br, bl) = box
                    
    result = []
    for p in range(min_pad, max_pad):
        cropped = crop(image,int(tl[1]),int(br[1]),int(tl[0]),int(tr[0]),p)
        border = 1
        cropped = cv2.copyMakeBorder(cropped, border, border, border, border, cv2.BORDER_CONSTANT,value=(200, 200, 200))
        result.append(resnet.classify(cropped,whitelist=whitelist) )
    return max(result,key=lambda x: x[1])[0]


def sort_boxes(boxes, is_horizontal=True):
    # 1 - sort by Y, 0 - sort by X
    sort_by = int(is_horizontal) 

    # sort all by Y/X
    boxes = sorted(boxes, key=lambda x: x[0][sort_by]) 

    # sort by height/width levels
    (tl, tr, br, bl) = boxes[0]
    level_min = min(tl[sort_by],bl[sort_by])
    level_max = max(tr[sort_by],br[sort_by])

    levels = []
    level = [boxes[0]]
    for box in boxes[1:]:
        (tl, tr, br, bl) = box
        
        if is_horizontal:
            center = br[sort_by]/2 + tl[sort_by]/2
        else:
            center = br[sort_by]/2 + bl[sort_by]/2

        if level_min < center < level_max:
            level.append(box)
        else:
            levels.append(level)
            level = [box]
  
        level_min = min(tl[sort_by],bl[sort_by])
        level_max = max(tr[sort_by],br[sort_by])
    levels.append(level)

    # sort each level by X/Y
    result = []
    for level in levels:
        result.extend(sorted(level, key=lambda x: x[0][1-sort_by]))

    return np.array(result)

def is_horizontal(detected_boxes):
    ratios = []
    r = []
    for box in detected_boxes:
        (tl, tr, br, bl) = box
        w = abs(br[0] - bl[0])
        h = abs(br[1] - tr[1])
        ratios.append(w/h)
        r.append(w/h > 1)
    return np.mean(ratios)>1 #np.mean(r) > 0.7


def limit_boxes(images_with_boxes):
    # limit boxes according to boxes_length
    boxes_length = [4,7,4]
    curr_len,box_count = 0, 0
    j = 0
    for i,(_, boxes) in enumerate(images_with_boxes):
        j = i
        curr_len += len(boxes)        
        if curr_len >= boxes_length[box_count]:
            curr_len = 0    
            box_count +=1        
        if box_count == len(boxes_length):
            break
    return images_with_boxes[:j+1]


def get_whitelist(box_num,total_len):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    digits = '0123456789'
    both = letters + digits
    whitelists = [letters] + [digits] * (total_len - 2) + [both]
    return whitelists[box_num]


result_regex = re.compile(r'[A-Z]{4}\d{5,7}([A-Z]|\d){1,4}') 
def is_valid(text, min_symbols=10, strict=False): 
    if len(text) > min_symbols:
        if strict:
            if result_regex.match(text):
                return True
        else:
            return True
    return False


if __name__ == "__main__":
    #5 11.jpeg 164 332 - v
    #t3 t4 t5 - h
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default='../data/test_dataset/images/12.png', #35.png
        help="path to input image to be OCR'd")
    args = vars(ap.parse_args())

    path = Path(args["image"])
    
    img = cv2.imread(str(path))
    result = detect(img,False)
    print(result)
    