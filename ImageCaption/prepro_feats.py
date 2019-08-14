from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
from multiprocessing import Pool
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import time
from Queue import Empty, Full, Queue
import threading
from threading import Thread

from dataloader_super import *
import opts
import misc.utils as utils

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet

def np_save(tmp) :
    np_save_(*tmp)

def np_save_(image_id, fc_feat, att_feat) :
    np.save(os.path.join(dir_fc, image_id), fc_feat)
    np.savez_compressed(os.path.join(dir_att, image_id), feat=att_feat)

class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for batch, data \
                in enumerate(dataset_generator):
            # We fill the queue with new fetched batch until we reach the max       size.
            batches_queue.put((batch, data) \
                              , block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, opt):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, data = batches_queue.get(block=True)
        data['img'] = np.stack(data['img'])
        cuda_batches_queue.put((batch, data), block=True)
        if tokill() == True:
            return



def main(opt):
    # Our train batches queue can hold at max 12 batches at any given time.
    # Once the queue is filled the queue is locked.
    train_batches_queue = Queue(maxsize=20)
    # Our numpy batches cuda transferer queue.
    # Once the queue is filled the queue is locked
    # We set maxsize to 3 due to GPU memory size limitations
    cuda_batches_queue = Queue(maxsize=10)
    dir_fc = 'data/%s/feat299_%s_fc'%(opt.dataset, opt.cnn_model)
    #dir_att = 'data/coco/feat299_resnet101_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    #if not os.path.isdir(dir_att):
    #    os.mkdir(dir_att)

    for split in ['all']:
        print('split', split)
        training_set_generator = InputGen(opt, split)
        opt.use_att = utils.if_use_att(opt.caption_model)
        opt.vocab_size = training_set_generator.vocab_size
        opt.seq_length = training_set_generator.seq_length

        train_thread_killer = thread_killer()
        train_thread_killer.set_tokill(False)
        preprocess_workers = 1

        # We launch 4 threads to do load &amp;&amp; pre-process the input images
        for _ in range(preprocess_workers):
            t = Thread(target=threaded_batches_feeder, \
                       args=(train_thread_killer, train_batches_queue, training_set_generator))
            t.start()
        
        cuda_transfers_thread_killer = thread_killer()
        cuda_transfers_thread_killer.set_tokill(False)
        cudathread = Thread(target=threaded_cuda_batches, \
                            args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue, opt))
        cudathread.start()


        cnn_model = utils.build_cnn(opt)
        cnn_model.cuda()
        cnn_model.eval()
        '''
        if opt.cnn_model == 'sceneprint':
            for layer in cnn_model.modules():
                if isinstance(layer, torch.nn.modules.BatchNorm2d):
                    print(layer)
                    layer.reset_running_stats()
        '''
        if opt.gpu_num > 1:
            cnn_model = torch.nn.DataParallel(cnn_model, device_ids=range(opt.gpu_num))
        print('sample counts : %d'%(training_set_generator.get_samples_count()))
        N = training_set_generator.get_samples_count() // opt.batch_size + 1
        with torch.no_grad():
            for i in range(N) :
                _, data = cuda_batches_queue.get(block=True)
                data['img'] = utils.prepro_images(data['img'], opt.img_csize, False)
                images = Variable(torch.from_numpy(data['img'])).cuda()
                att_feats = cnn_model(images).permute(0, 2, 3, 1)
                fc_feats = att_feats.mean(2).mean(1)
                for k in range(opt.batch_size) :
                    image_id = data['infos'][k]['image_id']
                    path_items = image_id.strip().split('/')
                    if len(path_items) > 1:
                        subdir_fc = os.path.join(dir_fc, path_items[0])
                        if not os.path.isdir(subdir_fc):
                            os.mkdir(subdir_fc)
                    #subdir_att = os.path.join(dir_att, image_id.strip().split('/')[0])
                    #if not os.path.isdir(subdir_att):
                    #    os.mkdir(subdir_att)
                    np.save(os.path.join(dir_fc, image_id), fc_feats[k].data.cpu().float().numpy())
                    #np.savez_compressed(os.path.join(dir_att, image_id), feat=att_feats[k].data.cpu().float().numpy())
                if i % 100 == 0:
                    print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
            
        train_thread_killer.set_tokill(True)
        cuda_transfers_thread_killer.set_tokill(True)
        for _ in range(preprocess_workers):
            try:
                # Enforcing thread shutdown
                train_batches_queue.get(block=True, timeout=1)
                cuda_batches_queue.get(block=True, timeout=1)
            except Empty:
                pass
       

start_time = time.time()
opt = opts.parse_opt()
main(opt)
print('total time: %.2f'%((time.time() - start_time)/3600))

