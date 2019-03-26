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

etc = {
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

etc1_2 = {
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

etc2_3 = {  #2_*
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

etc3 = {
    'method': 'order',
    'margin': 0.05,
    'abs': False,
}

etc4 = {
    'method': 'order',
    'margin': 0.05,
    'abs': True,
}

etc2_hyper_m0p1_1 = {
    'method': 'order',
    'margin': 0.1,
    'abs': True,
}

etc2_hyper_m0p1_2 = {
    'method': 'order',
    'margin': 0.1,
    'abs': True,
}

etc2_hyper_m0p03_1 = {
    'method': 'order',
    'margin': 0.03,
    'abs': True,
}

etc2_hyper_m0p03_2 = {
    'method': 'order',
    'margin': 0.03,
    'abs': True,
}


default_params = {
    'max_epochs': 100,
    'data': 'coco',
    'cnn': '10crop',
    'dim_image': 4096,
    'encoder': 'gru',
    'dispFreq': 10,
    'grad_clip': 2.,
    'optimizer': 'adam',
    'batch_size': 128,
    'dim': 1024,
    'dim_word': 300,
    'lrate': 0.001,
    'validFreq': 300
    #'v_norm': 'l2'
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['order', 'symmetric', 'etc2_hyper_m0p03_1', 'etc2_hyper_m0p03_2'])
args = parser.parse_args()
model_params = eval(args.model)

model_params.update(default_params)

name = args.model
train.trainer(name=name, **model_params)

'''
model_params = eval('order')
model_params.update(default_params)

for i in range(2,6):
    name = 'order%d'%(i)
    train.trainer(name=name, **model_params)
'''


