#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import matplotlib as mpl
mpl.use('Agg')

sys.path.insert(0, '/home1/caffe/caffe/' + 'python')
import caffe
from caffe.proto import caffe_pb2
import lmdb
import cv2
import time
import numpy as np
import random
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import time

deployProto_path='fdy_zg_c12_inside_VGG16_dmax.prototxt'
caffe_model_path='dmax_iter_129000.caffemodel'
testImgTxt_path='/home2/project_data/BreakPaper/FDY_POY_cam12_inside/testData.txt'

#初始化网络参数
# net = caffe.Net(deployProto_path, caffe_model_path, caffe.TEST)
#
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2, 1, 0))

#返回pair对的左边图像list
def readTxtTopath(testImgTxt_path):
    f=open(testImgTxt_path,'r')
    lines=f.readlines()

    left_list=[]
    left_labels=[]
    for line in lines:
        line=line.strip()
        sample_path=line.split(' ')[0]
        sample_label=int(line.split(' ')[1])
        left_list.append(sample_path)
        left_labels.append(sample_label)
    return left_list,left_labels

def getSingleImgProb(img):
    net = init_net()
    im_data=caffe.io.load_image(img)
    net.blobs['data'].data[...]=net.transformer.preprocess('data',im_data)

    #执行前向
    out=net.forward()
    prob=net.blobs['prob'].data[0].flatten()
    return prob

def getBatchImagProb(imgList):
    net = init_net()
    Batch_size=net.blobs['data'].data.shape[0]
    Batch_num=int(math.ceil(float(len(imgList))/Batch_size))
    cur=0
    prob_out=[]

    for i in range(Batch_num):
       # print "i=",i
        caffe_in = np.zeros(net.blobs['data'].data.shape, dtype=np.float32)
        for j in range(Batch_size):
            #print j
            if cur==len(imgList):
                continue
            cur_img=caffe.io.load_image(imgList[cur])
            caffe_in[j]=net.transformer.preprocess('data',cur_img)
            cur+=1
        # 执行前向
        out = net.forward_all(**{net.inputs[0]: caffe_in})
        batch_prob = out[net.outputs[0]]
       # print(batch_prob)
        prob_out.append(batch_prob)

        if cur==len(imgList):
            break
    prob_out=np.array(prob_out)
    #print prob_out
    return prob_out

def drawROCcurve(gtLabels,predicProbs):
    fpr,tpr,threshold=roc_curve(gtLabels,predicProbs)
    roc_auc=auc(fpr,tpr)

    plt.figure()
    lw=2
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("./test.png")


def extract_feat(images,blobname,shape):
    '''
    faces: num * height * width * channel
    blobname: e.g. 'fc6', 'pool5' ...
    shape: shape for this blob, e.g fc6: [4096] , pool5:[512,7,7]
    '''
    net = init_net()
    num = len(images)
    batch_size=128 # 每次处理的数量
    iters = num/batch_size # 全部图片一共要处理多少次（iters是循环的意思，用词貌似不恰当，然而习惯了-_-||）
    mod = np.mod(num,batch_size) # 余数
    netdata = np.empty([num,shape])
    #print netdata.shape
    if mod!=0:
        iters += 1 # 注意，假如batch不能整除num，最后需要再添加一次
    for i in range(iters):
        print('forward process %d/%d'%(i+1,iters))
        idx0 = i*batch_size
        if i==(iters-1) and mod!=0:
            idx1 = i*batch_size+mod # 注意，假如batch不能整除num，最后处理的数量就不是batch_size而是mod了
        else:
            idx1 = (i+1)*batch_size
        data_batch = images[idx0:idx1] # 选取从idx0到idx1之间的数据，python中的下标规则是，假如a=[1,2,3]，那么a[0:2]=[ a[0],a[1] ] = [1,2](随便写的，不要计较语法...)
        netdata_batch = pro_batch(data_batch,net,blobname,shape) # 处理batch数据
        netdata[idx0:idx1]=netdata_batch
    return netdata

def pro_batch(image_list,net,blobname,shape):
    '''
    faces: num * height * width * channel
    blobname: e.g. 'fc6', 'pool5' ...
    shape: shape for this blob, e.g fc6: [4096] , pool5:[512,7,7]
    '''
    images = prepare_data(image_list)
    netdata_batch = net.forward_all( data = images,blobs=[blobname] )[blobname]
    return netdata_batch

def init_net():
    #caffe.set_device(0)
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(deployProto_path, caffe_model_path, caffe.TEST)
    return net

def prepare_data(image_list):
    images_resize=np.empty([len(image_list),3,40,200])
    for i in range(0,len(image_list)):
        image=cv2.imread(image_list[i],1)
        #image=image0[2:98,2:98]  #python 矩阵计数包头不包尾
        #image=cv2.resize(image,(96,96))
        images_resize[i]=image.transpose((2,0,1))
    return images_resize

if __name__=="__main__":
    start = time.time()
    leftFiles,leftLabels=readTxtTopath(testImgTxt_path)
    leftLabels=np.array(leftLabels)
    leftLabels[leftLabels<3]=0
    leftLabels[leftLabels==3]=1
    leftLabels[leftLabels==4]=1
    leftLabels[leftLabels==5]=0
    leftLabels[leftLabels>5]=1
    #np.save('groudtruth_labels.npy',np.array(leftLabels))
    #leftLabels=np.load('groudtruth_labels.npy')

    used_prob=np.empty([len(leftLabels),4])
    leftProbs=extract_feat(leftFiles,'prob',8)
    used_prob[:,0:2]=leftProbs[:,3:5]
    used_prob[:,2:4]=leftProbs[:,6:8]
    
    np.save('vgg_iter_dmax_all_prob.npy',np.array(leftProbs))
    pos_probs=np.empty(leftProbs.shape[0])
    for i in range(0,leftProbs.shape[0]):
        pos_probi=np.max(used_prob[i,:])#np.max(leftProbs[i,4:8])
        pos_probs[i]=pos_probi
    np.save('vgg_iter_dmax_used_prob.npy',pos_probs)
    #pos_probs=np.load('vgg_iter_online_used_prob.npy')
    
    drawROCcurve(np.array(leftLabels),pos_probs)
    print("Done in %.2f s." % (time.time() - start))
    
    