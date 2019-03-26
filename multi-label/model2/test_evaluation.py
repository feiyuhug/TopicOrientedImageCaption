# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:07:34 2017

@author: shouw
"""
cd '../caffe'
import numpy as np
import sys
caffe_root='../caffe'
sys.path.append(caffe_root+'/python')
import caffe
# from matplotlib import pyplot as plt
# %matplotlib inline

# caffe_root='../caffe/'
# import sys
# sys.path.insert(0,caffe_root+'python')
# import caffe
# import numpy as np


def sigmoid(x):
    return 1.0/(1+np.exp(-x))
caffe.set_mode_gpu() #caffe.set_mode_cpu()
caffe.set_device(1)
# caffe_model=caffe_root+'huawei_set/snapshot1/voc_snapshot_iter_4000.caffemodel'
caffe_model=caffe_root+'../train.caffemodel'
net=caffe.Net(caffe_root+'../model.prototxt',caffe_model,caffe.TEST)
all_samples=3620  #SCENE_SET(test_all_samples); VOC2007(4952)
n_class=12   #SCENE_SET(12); VOC2007(20)
batch_size=net.blobs['label'].data.shape[0]
test_iters=int(all_samples/batch_size)+1

#APC
Loss=0
Accuracy=0
for k in xrange(test_iters):
    net.forward()
    Loss+=net.blobs['loss'].data
    fea=net.blobs['fc8a_h'].data
    predict=sigmoid(fea)>0.5
    targetlabel=net.blobs['label'].data.reshape(net.blobs['label'].data.shape[0],net.blobs['label'].data.shape[1])
    target=targetlabel>0
    nums,numf=predict.shape
    matcount=0
    for j in xrange(nums):
        if sum(targetlabel[j])!=0:
            ptrue_index=np.array([i for i in range(numf) if predict[j,i]==True])
            ttrue_index=np.array([i for i in range(numf) if target[j,i]==True])
            if set(ptrue_index)&set(ttrue_index)!=set([]):
                matcount+=1
        else:
            if sum(predict[j])==0:
                matcount+=1
    acc=float(matcount)/nums
    Accuracy+=acc
Loss/=test_iters
Accuracy/=test_iters
print("Loss: {:.4f}".format(Loss))
print("Accuracy: {:.4f}".format(Accuracy))

#AAC
Loss=0
Accuracy=0
for k in xrange(test_iters):
    net.forward()
    Loss+=net.blobs['loss'].data
    fea=net.blobs['fc8a_h'].data
    predict=sigmoid(fea)>0.5
    target=net.blobs['label'].data.reshape(net.blobs['label'].data.shape[0],net.blobs['label'].data.shape[1])>0
    nums,numf=predict.shape
    matcount=0
    for j in xrange(nums):
        if np.all(predict[j]==target[j])==True:
            matcount+=1
    acc=float(matcount)/nums
    Accuracy+=acc
Loss/=test_iters
Accuracy/=test_iters
print("Loss: {:.4f}".format(Loss))
print("Accuracy: {:.4f}".format(Accuracy))

#AP&mAP
label=np.zeros((1,n_class),dtype=int)
out=np.zeros((1,n_class),dtype='float32')
Loss=0
for k in xrange(test_iters):
    net.forward()
    Loss+=net.blobs['loss'].data
    fea=net.blobs['fc8a_h'].data
    fea=sigmoid(fea)
    batch_label=net.blobs['label'].data.reshape(net.blobs['label'].data.shape[0],net.blobs['label'].data.shape[1]).astype('int')
    label=np.concatenate([label,batch_label],axis=0)
    out=np.concatenate([out,fea],axis=0)
label=np.delete(label,0,0)
out=np.delete(out,0,0)
out=out[:all_samples]
label=label[:all_samples]
order_index=np.argsort(-out,axis=0)
AP=list()
for i in xrange(label.shape[1]):
    order_label=label[order_index[:,i],i]
    tp=(order_label>0).astype('int')
    fp=(order_label==0).astype('int')
    tp=np.cumsum(tp)
    fp=np.cumsum(fp)
    Recall=tp.astype('float')/sum(order_label>0)
    Precision=tp.astype('float')/(fp+tp)
    ap=0
    for t in np.arange(0,1.1,0.1):
        p=np.max(Precision[Recall>=t])
        if p==None:
            p=0
        ap=ap+p/11
    AP.append(ap)
Loss/=test_iters
if n_class==12:
	txtlist=['indoor','person','LDR','green','mall','beach','back','sunset','blue','snow','night','text']
else:
	txtlist=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',\
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
print("Loss: {:.4f}".format(Loss))
for i in xrange(len(txtlist)):
    print("class:%s,ap=%10.3f" %(txtlist[i],AP[i]))
mAP = np.mean(AP)
print("mAP: {:.3f}".format(mAP))