from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from PIL import Image, ImageDraw


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

        
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable  
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, orig_im, dim

def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp

def prep_image_annotation(img, inp_dim, annotation):
    '''
    功能：将输入图片、标注转换为网络需要的形式（对图片进行resize，使用的resize方式为等比例缩放，不足处补128）
    输入参数: img:待转换图片
             inp_dim:网络需要的输入大小[w, h]
             annotation:图片的原始标注
    输出参数：img_: torch.tensor，转换后的图片，大小为[1, channel, w, h]
             bbox: torch.tenosr，转换后的标注，大小为[1, 4]
    '''
    orig_img = cv2.imread(img)
    img_w, img_h = orig_img.shape[1], orig_img.shape[0]
    w, h = inp_dim

    # 对图片进行resize，比例即为 min(w/img_w, h/img_h)
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(orig_img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    # canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    # 转换为tensor，并进行归一化、补上第0维，[1, channel, w, h]
    img_ = canvas[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    # 对标注进行resize，即首先进行比例变换再加上偏移量, (w, h)无偏移
    origin_bbox_x, origin_bbox_y, origin_bbox_w, origin_bbox_h = annotation[0], annotation[1], annotation[2], annotation[3]
    bbox_x = int(origin_bbox_x * min(w/img_w, h/img_h)) + (w-new_w)//2
    bbox_y = int(origin_bbox_y * min(w/img_w, h/img_h)) + (h-new_h)//2
    bbox_w = int(origin_bbox_w * min(w/img_w, h/img_h))
    bbox_h = int(origin_bbox_h * min(w/img_w, h/img_h))

    # 转换为tensor，补上第0维，[1, 4]
    bbox = torch.tensor([bbox_x, bbox_y, bbox_w, bbox_h]).unsqueeze(0)

    return img_, bbox

if __name__ == "__main__":
    img = "C:\\Users\\Xiang\\Desktop\\2012_004331.jpg"
    a, b = prep_image_annotation(img, (416, 416), (155, 127, 106, 205)) 
    print(a.size(), b.size())
    print("end")
    pass
