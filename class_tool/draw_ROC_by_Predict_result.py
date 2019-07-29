#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

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
    plt.savefig("/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf2/roc/test.png")


def drawROCcurve2(gtLabels, predicProbs1, predicProbs2):
    fpr1, tpr1, threshold1 = roc_curve(gtLabels, predicProbs1)
    roc_auc1 = auc(fpr1, tpr1)

    # 画第一幅图
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='Online_ROC curve (area = %0.3f)' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.show()

    # 画第二幅图
    fpr2, tpr2, threshold2 = roc_curve(gtLabels, predicProbs2)
    roc_auc2 = auc(fpr2, tpr2)
    lw = 2
    plt.plot(fpr2, tpr2, color='black', lw=lw, label='new_ROC curve (area = %0.3f)' % roc_auc2)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("./compared_ROC.png")

def getTprBythresh(tpr1,fpr1,tpr2,fpr2,threshold1,threshold2,thresh):
    tpr1_tmp=[]
    tpr2_tmp=[]
    fpr1_tmp=[]
    fpr2_tmp=[]
    for i in range(0,len(threshold1)):
        if threshold1[i]>=thresh-thresh/100 and threshold1[i]<=thresh+thresh/100:
            tpr1_tmp.append(tpr1[i])
            fpr1_tmp.append(fpr1[i])
            
    for i in range(0,len(threshold2)):
        if threshold2[i]>=thresh-thresh/100 and threshold2[i]<=thresh+thresh/100:
            tpr2_tmp.append(tpr2[i])
            fpr2_tmp.append(fpr2[i])
            
    mtpr1=np.max(tpr1_tmp)
    mtpr2=np.max(tpr2_tmp)
    mfpr1=np.max(fpr1_tmp)
    mfpr2=np.max(fpr2_tmp)
    
    return mtpr1,mtpr2,mfpr1,mfpr2
    
def getThreshByfpr(threshold1,threshold2,fpr1,fpr2,fpr):
    out_thresh_1=[]
    for i in range(0,len(fpr1)):
        if fpr1[i]>=fpr-fpr/10 and fpr1[i]<fpr+fpr/10:
            out_thresh_1.append(threshold1[i])    
    
    
    out_thresh_2=[]
    for i in range(0,len(fpr2)):
        if fpr2[i]>=fpr-fpr/10 and fpr2[i]<fpr+fpr/10:
            out_thresh_2.append(threshold2[i]) 
    
    out_th1=np.mean(out_thresh_1)
    out_th2=np.mean(out_thresh_2)
    
    return out_th1,out_th2
    
def getThreshBytpr(threshold1,threshold2,tpr1,tpr2,tpr):
    out_thresh_1=[]
    for i in range(0,len(tpr1)):
        if tpr1[i]>=tpr-tpr/10 and tpr1[i]<tpr+tpr/10:
            out_thresh_1.append(threshold1[i])    

    out_thresh_2=[]
    for i in range(0,len(tpr2)):
        if tpr2[i]>=tpr-tpr/10 and tpr2[i]<tpr+tpr/10:
            out_thresh_2.append(threshold2[i]) 
    
    out_th1=np.mean(out_thresh_1)
    out_th2=np.mean(out_thresh_2)
    
    return out_th1,out_th2
    
def drawSemiLogxCurve(gtLabels, predicProbs1, predicProbs2):
    fpr1, tpr1, threshold1 = roc_curve(gtLabels, predicProbs1)
    roc_auc1=auc(fpr1,tpr1)

    #print 'tpr=',tpr1,'fpr=',fpr1
    fpr2, tpr2, threshold2 = roc_curve(gtLabels, predicProbs2)
    roc_auc2=auc(fpr2,tpr2)

    # print 'fpr1=', fpr1
    # print 'fpr2=', fpr2
    thresh=0.125
    [mtpr1,mtpr2,mfpr1,mfpr2]=getTprBythresh(tpr1,fpr1,tpr2,fpr2,threshold1,threshold2,thresh)
    print('thresh=',thresh,'softmax_mtpr1=',mtpr1,'LGM_mtpr2=',mtpr2,'softmax_mfpr1=',mfpr1,'LGM_mfpr2=',mfpr2)
    fpr=mfpr1
    [ut_th1,out_th2]=getThreshByfpr(threshold1,threshold2,fpr1,fpr2,fpr)
    print('fpr=',fpr,'softmax_thresh=',ut_th1,'LGM_thresh=',out_th2)
    fpr=mfpr2
    [ut_th1,out_th2]=getThreshByfpr(threshold1,threshold2,fpr1,fpr2,fpr)
    print('fpr=',fpr,'softmax_thresh=',ut_th1,'LGM_thresh=',out_th2)

    # 画第一幅图
    plt.figure()
    lw = 2
    #plt.figure(figsize=(10, 10))
    plt.semilogx(fpr1, tpr1, color='darkorange', lw=lw, label='vgg16_prob_ROC_curve (area = %0.3f)'% roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")

    # 画第二幅图
    lw = 2
    plt.semilogx(fpr2, tpr2, color='black', lw=lw, label='vgg16_lpf2_prob_ROC_curve (area = %0.3f)'% roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()
    plt.savefig("/home1/pytorch_code/antialiased-cnns/weights/compared_semilogx.png")

def drawSemiLogxCurve4(gtLabels, predicProbs1, predicProbs2, predicProbs3, predicProbs4):
    fpr1, tpr1, threshold1 = roc_curve(gtLabels, predicProbs1)
    roc_auc1=auc(fpr1,tpr1)

    #print 'tpr=',tpr1,'fpr=',fpr1
    fpr2, tpr2, threshold2 = roc_curve(gtLabels, predicProbs2)
    roc_auc2=auc(fpr2,tpr2)
    
    fpr3, tpr3, threshold3 = roc_curve(gtLabels, predicProbs3)
    roc_auc3=auc(fpr3,tpr3)
    
    fpr4, tpr4, threshold4 = roc_curve(gtLabels, predicProbs4)
    roc_auc4=auc(fpr4,tpr4)

    # print 'fpr1=', fpr1
    # print 'fpr2=', fpr2
    # thresh=0.125
    # [mtpr1,mtpr2,mfpr1,mfpr2]=getTprBythresh(tpr1,fpr1,tpr2,fpr2,threshold1,threshold2,thresh)
    # print('thresh=',thresh,'softmax_mtpr1=',mtpr1,'LGM_mtpr2=',mtpr2,'softmax_mfpr1=',mfpr1,'LGM_mfpr2=',mfpr2)
    # fpr=mfpr1
    # [ut_th1,out_th2]=getThreshByfpr(threshold1,threshold2,fpr1,fpr2,fpr)
    # print('fpr=',fpr,'softmax_thresh=',ut_th1,'LGM_thresh=',out_th2)
    # fpr=mfpr2
    # [ut_th1,out_th2]=getThreshByfpr(threshold1,threshold2,fpr1,fpr2,fpr)
    # print('fpr=',fpr,'softmax_thresh=',ut_th1,'LGM_thresh=',out_th2)

    # 画第一幅图
    plt.figure()
    lw = 2
    #plt.figure(figsize=(10, 10))
    plt.semilogx(fpr1, tpr1, color='darkorange', lw=lw, label='vgg16_prob_ROC_curve (area = %0.3f)'% roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")

    # 画第二幅图
    lw = 2
    plt.semilogx(fpr2, tpr2, color='black', lw=lw, label='vgg16_lpf2_prob_ROC_curve (area = %0.3f)'% roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    lw = 2
    plt.semilogx(fpr3, tpr3, color='blue', lw=lw, label='vgg16_lpf3_prob_ROC_curve (area = %0.3f)'% roc_auc3)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    lw = 2
    plt.semilogx(fpr4, tpr4, color='red', lw=lw, label='vgg16_lpf5_prob_ROC_curve (area = %0.3f)'% roc_auc4)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()
    plt.savefig("/home1/pytorch_code/antialiased-cnns/weights/compared_semilogx.png")

 
def Visualize_class(gtLabels,predicProbs):
    f = plt.figure(figsize=(16, 9))
    c = ['#ff0000', '#ffff00']
    for i in range(2):
        plt.plot(predicProbs[gtLabels == i], predicProbs[gtLabels == i], '.', c=c[i])
    plt.legend(['0', '1'])
    plt.grid()
    plt.savefig("./visual_1.png")


if __name__ == "__main__":
    gtLabel = np.load('/home1/pytorch_code/antialiased-cnns/weights/vgg16/roc/gt_label.npy')
    vgg16_prob = np.load('/home1/pytorch_code/antialiased-cnns/weights/vgg16/roc/tested_used_prob.npy')
    vgg16_lpf2_prob = np.load('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf2/roc/tested_used_prob.npy')
    vgg16_lpf3_prob = np.load('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf3/roc/tested_used_prob.npy')
    vgg16_lpf5_prob = np.load('/home1/pytorch_code/antialiased-cnns/weights/vgg16_lpf5/roc/tested_used_prob.npy')
    
    # print gtLabel.shape,onlineProbs.shape,predicProbs.shape
    # print("gtlabel:",gtLabel)
    # print("predicProbs:",predicProbs)
    
    #drawROCcurve(gtLabel, vgg16_lpf2_prob)
    drawSemiLogxCurve4(gtLabel, vgg16_prob, vgg16_lpf2_prob, vgg16_lpf3_prob, vgg16_lpf5_prob)
    #Visualize_class(gtLabel, res18)















