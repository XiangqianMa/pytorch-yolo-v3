
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from bbox import bbox_iou

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    功能：该函数在Detection类的forward()函数中被调用，作用是对网络输出的特征图进行处理，依据anchor boxes等参数得到YOLO网络的预测。
    输入参数:prediction：网络的输出特征图
            inp_dim: 输入图像的尺度
            anchors：预设的先验框
            num_classes：类别数目
            CUDA：是否使用cuda，默认为True
    输出: 网络的输出（prediction）
    '''
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2) # 计算出从输入图片到输出特征图总共缩小了多少倍
    grid_size = inp_dim // stride # 计算出grid的大小
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # print("prediction size: ", prediction.size())
    # print("batch_size: {}, bbox_attrs: {}, num_anchors: {}, grid_size: {}".format(batch_size, bbox_attrs, num_anchors, grid_size))

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()# pytorch的view只能作用于一整块内存上，如果在view之前进行了transpose、permute等操作，需要先调用contiguous将tensor置于一整块内存中
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    # centre_X, centre_Y以及object confidence都是位于[0,1]的数，因而使用sigmoid函数对其范围进行限制
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    # 给预测的中心位置加上偏置
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
   
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride # 将特征图上的预测转换为输入图片上的预测，乘以stride即可（二维张量） 
    
    return prediction

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    '''
    对网络的输出进行处理得到最终的输出
    '''
    # 将小于置信度阈值的边界框的整行设置为0
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    # 将bboxes的坐标转换为对角线坐标的形式
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    # 对每张图片依次进行置信度阈值判断和NMS
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        
        # 得到具有最大分数的类别，以及该类别的索引
        # 将num_classes个类别分数删除
        # 加入具有最大分数的类别的索引与分数
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1) # 在索引的第二维新加一维 
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        # 去除置信度分数为0的边界框（即行）
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        # 按照类别进行NMS
        for cls in img_classes:
            # 提取特定类的检测结果
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # 对同一类的边界框进行排序，具有最大目标置信度的排在顶部
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    # 计算i索引的边界框与i之后的所有边界框的IoU
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    # 将IoU大于阈值的所有边界框清零
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""

def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda().half()
        y_offset = y_offset.cuda().half()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.HalfTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:,:, 5 : 5 + num_classes])).data

    prediction[:,:,:4] *= stride
    
    
    return prediction


def write_results_half(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).half().unsqueeze(2)
    prediction = prediction*conf_mask
    
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half()
        
        
        
                
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind]

        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind]
                    
                    
            
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output
