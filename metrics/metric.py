import os
import math
import random
import numpy as np
import torch
import cv2
from PIL import Image
from PIL import ImageOps
import re


def calculate_ssim(img1, img2, border=0): ### calculate the ssim based on two images
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2] ##(h,w,c)
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i])) ## calculate the ssim based on  each channel and then average over channel
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    
def calculate_PSNR(pred, gt, shave_border=0, max_pixel_value=255.0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = np.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * np.log10(max_pixel_value / rmse)

### base function ###
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
###
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def lstFiles(path):
    image_list=[]
    for image in os.listdir(path):
        if ".png" in image.lower():
            image_list.append(os.path.join(path, image))
        elif ".jpeg" in image.lower():
            image_list.append(os.path.join(path, image))
    image_list = sorted(image_list, key=numericalSort)
    return image_list

def Normalization(image): ## normalize range to 0-255 for ssim calculateion need to be array
    image = image.astype(np.float64)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val) * 255.0
    else:
        image = np.zeros_like(image)
    return image

## calculate the ssim based on two path 
class calculate_metrics_folder(object):
    def __init__(self, input_path, target_path, target_size =(256,256)):
        self.input_path =input_path ## the path to directory
        self.target_path = target_path
        self.size = target_size
    
    def read_image(self, path):
        try:
            target_size = self.size
            readed_image = Image.open(path)
            readed_image = readed_image.convert("RGB")
            original_size = readed_image.size
            if original_size[0] < target_size[0] or original_size[1] < target_size[1]:
                delta_width = max(target_size[0] - original_size[0], 0)
                delta_height = max(target_size[1] - original_size[1], 0)
                padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
                image = ImageOps.expand(readed_image, padding)
            else:
                image = readed_image.resize(target_size, Image.Resampling.LANCZOS)
            array_image = np.array(image).astype(np.float64)
            return array_image
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            return None
    
    def metrics_folder(self):
        input_list = os.listdir(self.input_path)
        # target_list = os.listdir(self.target_path)
        ssim_result =[]
        psnr_result = []
        for image_path in input_list:

            input_image = self.read_image(os.path.join(self.input_path,image_path))
            target_image = self.read_image(os.path.join(self.target_path,image_path))

            input_image = Normalization(input_image) #(H,W,C) np.float64 array  0-255
            target_image = Normalization(target_image)
  
            ssim_result.append(calculate_ssim(input_image,target_image))
            psnr_result.append(calculate_PSNR(input_image,target_image))

            if np.isnan(np.sum(ssim_result)) or np.isinf(np.sum(ssim_result)) or np.isnan(np.sum(psnr_result)) or np.isinf(np.sum(psnr_result)):
                raise ValueError("Invalid SSIM or PSNR results encountered (NaN or Inf)")

        return (np.sum(ssim_result)/len(ssim_result),np.sum(psnr_result)/len(psnr_result))


   