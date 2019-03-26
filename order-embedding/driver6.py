import train

order = {
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

symmetric = {
    'method': 'cosine',
    'margin': 0.2,
    'abs': False,
}

default_params = {
    'max_epochs': 100,
    'data': 'coco',
    'cnn': 'finetuned_t200_10crop_lmdb1_bak_resnet',
    'dim_image': 2048,
    'dim_topic': 200,
    'use_topic': False,
    'encoder': 'gru',
    'dispFreq': 10,
    'grad_clip': 2.,
    'optimizer': 'adam',
    'batch_size': 128,
    'dim': 1024,
    'dim_word': 300,
    'lrate': 0.001,
    #'lrate': 0.0001,
    'validFreq': 300
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['order', 'symmetric'])
parser.add_argument('name')
args = parser.parse_args()
model_params = eval(args.model)

model_params.update(default_params)

name = args.name
#load_from = 'snapshots/sentence-image-finetuned-t200-lmdb1-topic-insert_0.2_order_1'
load_from = None
train.trainer(load_from=load_from, name=name, **model_params)


