import numpy as np
from matplotlib import pyplot as plt
import lmdb
caffe_root='/home/wujian/yuniange/caffe/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import cv2

for dataset in ['test', 'val', 'train'] :
    '''
    label = np.load('../code/nmf/flickr30k/doc-topic_%s_t100_bi.npy'%(dataset))
    label=label.reshape((label.shape[0],label.shape[1],1,1))
    map_size1=label.nbytes*10
    N1=range(label.shape[0])
    env=lmdb.open('flickr30k/%s_label_lmdb'%(dataset),map_size=map_size1)
    with env.begin(write=True) as txn:
        for i in N1:
            datum=caffe.io.array_to_datum(label[i])
            str_id='{:08}'.format(i)
            txn.put(str_id.encode('ascii'),datum.SerializeToString())
    
    '''
    img_ids = []
    with open('../data/flickr30k/splits/kar_%s.ids'%(dataset)) as f:
        for line in f:
            img_ids.append(int(line.strip()))

    rootname = '../data/flickr30k/images/'
    #map_size=len(img_ids)*3*256*256*4*10
    map_size=len(img_ids)*3*256*500*4*10
    env=lmdb.open('flickr30k/%s_lmdb2'%(dataset),map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(len(img_ids)):
            if i % 1000 == 0:
                print i
            #img = cv2.imread(rootname + '%02d/%d.jpg'%(img_ids[i]/10000, img_ids[i]))
            img = cv2.imread(rootname + '%d.jpg'%(img_ids[i]))
            min_l, max_l = min([img.shape[0], img.shape[1]]), max([img.shape[0], img.shape[1]])
            rescale_ratio = 256.0/min_l
            img = cv2.resize(img, (int(img.shape[1]*rescale_ratio), int(img.shape[0]*rescale_ratio)))
            #img = cv2.resize(img, (256, 256))
            img = img.transpose(2, 0, 1)        

            datum=caffe.proto.caffe_pb2.Datum()
            datum.channels=img.shape[0]
            datum.height=img.shape[1]
            datum.width=img.shape[2]
            datum.data=img.tobytes()
            str_id='{:08}'.format(i)
            txn.put(str_id.encode('ascii'),datum.SerializeToString())  
    
    

