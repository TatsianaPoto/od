### Useful links
# https://www.geeksforgeeks.org/detect-and-recognize-car-license-plate-from-a-video-in-real-time/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html


import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure
import imutils


def sort_cont(character_contours):
    """ 
    To sort contours 
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]

    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours,
                                                          boundingBoxes),
                                                      key=lambda b: b[1][i],
                                                      reverse=False))

    return character_contours


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def canny(img):
    thresh = auto_canny(img)

    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    viz = img.copy()
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours = []
    for i, c in enumerate(sorted_ctrs):
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(img.shape[0])
        
        keepAspectRatio = aspectRatio < 1.0
        
        if keepAspectRatio and boxW > 5:
            
            hull = cv2.convexHull(c)
            cv2.drawContours(viz, [hull], -1, 255, -1)
            
            # cv2.rectangle(viz, (boxX,boxY),(boxX+boxW, boxY+boxH),(0,0,255),1)


            contours.append(c)
    cv2.imshow('canny',viz)
    cv2.waitKey()
            
    return contours


def connected_components(img):
    thresh = cv2.adaptiveThreshold(img.astype(np.uint8), 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11, 2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
    # thresh = cv2.morphologyEx(src=thresh,
    #                     op=cv2.MORPH_CLOSE,
    #                     kernel=kernel)

    # thresh = cv2.bitwise_not(thresh)
    labels = measure.label( thresh, background=0) #,

    charCandidates = img.copy()
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts,_ = cv2.findContours(labelMask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(img.shape[0])

            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = True #20 < boxH < 80 #heightRatio > 0.5 and heightRatio < 0.95

            if keepAspectRatio and keepHeight and boxW > 5:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    cv2.imshow('cc',charCandidates)
    cv2.waitKey()

    contours, hier = cv2.findContours(charCandidates,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    return contours


def segment_chars(plate_img, fixed_width):
    """ 
    extract Value channel from the HSV format 
    of image and apply adaptive thresholding 
    to reveal the characters on the license plate 
    """
    imgBlurred = cv2.GaussianBlur(plate_img, (3, 3), 0)
    V = cv2.split(cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2HSV))[2]

    # V = cv2.Laplacian(V,cv2.CV_64F,ksize=1)

    contours = connected_components(V)
    contours = canny(V)
    # cv2.imshow('image',thresh)
    # cv2.waitKey()

    print(len(contours))
    characters = []
    if contours:
        contours = sort_cont(contours)

        # value to be added to each dimension
        # of the character
        addPixel = 4
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            x1 = int(max(0,x/1 -addPixel))
            y1 = int(max(0,y/1 -addPixel))
            
            x2 = int(min(x1 + w/1 + (addPixel * 2), plate_img.shape[1]))
            y2 = int(min(y1 + h/1 + (addPixel * 2), plate_img.shape[0]))
                    
            characters.append(plate_img[y1:y2,x1:x2])

        return characters
    else:
        return None