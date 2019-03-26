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
    'data': 'flickr30k',
    'cnn': 'coco-t100-lmdb2',
    'dim_image': 4096,
    'dim_topic': 100,
    'encoder': 'gru',
    'dispFreq': 10,
    'grad_clip': 2.,
    'optimizer': 'adam',
    'batch_size': 128,
    'dim': 1024,
    'dim_word': 300,
    'lrate': 0.001,
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
train.trainer(name=name, **model_params)


