### Useful links
# https://docs.opencv.org/master/d1/dfd/tutorial_motion_deblur_filter.html
# https://pylops.readthedocs.io/en/latest/tutorials/deblurring.html
# http://amroamroamro.github.io/mexopencv/opencv/weiner_deconvolution_demo_gui.html
# https://github.com/KupynOrest/DeblurGAN
# https://www.programmersought.com/article/5311463439/


import cv2
import math
import numpy as np
from numpy import fft

from scipy.signal import convolve2d as conv2
from skimage import restoration

from scipy.signal.signaltools import wiener

def get_motion_psf(image_size, motion_angle, motion_dis):
    PSF = np.zeros(image_size) #point spread function
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2
 
    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)
 
    # Set the motion_dis points to 1 in the corresponding angle
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1
 
    return PSF / PSF.sum() # normalization

def get_wiener(input, PSF, eps, SNR=0.001): # Filter, SNR=0.01
    input_fft=fft.fft2(input)
    PSF_fft=fft.fft2(PSF) +eps
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + SNR)
    result=fft.ifft2(input_fft * PSF_fft_1)
    result=np.abs(fft.fftshift(result))
    return result

def inverse(input, PSF, eps):       
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps #noise power
    result = fft.ifft2(input_fft / PSF_fft) #Calculate the inverse Fourier transform of F(u,v)
    result = np.abs(fft.fftshift(result))
    return result


image = cv2.imread('blurred.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('orig', image_gray)
cv2.waitKey()


# filtered_img = wiener(image_gray, (2, 2)) 
# cv2.imshow('deblurred', filtered_img.astype(np.uint8))
# cv2.waitKey()

psf = get_motion_psf(image_gray.shape, 2, 5)

norm_psf = cv2.normalize(psf, None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('psf', (norm_psf*255).astype(np.uint8))
cv2.waitKey()

#norm_img = cv2.normalize(image_gray, None,0.0, 1.0, cv2.NORM_MINMAX)


result=get_wiener(image_gray,psf,1e-5) 

norm_result = cv2.normalize(result, None,0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('deblurred', (norm_result * 255).astype(np.uint8))
cv2.waitKey()
