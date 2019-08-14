# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import argparse
from random import shuffle, seed
import string
import numpy as np

def main(params) :
    pkg = json.load(open(params['input_json'], 'r'))

    itow = pkg['ix_to_word']

    output = {}
    output['ix_to_word'] = itow
    output['images'] = []

    N=0
    with open(params['input_txt'], 'r') as f :
        for filename in f :
            image_id = filename.strip()
            jimg = {}
            jimg['image_id'] = image_id
            jimg['id'] = N
            jimg['split']= params['split']
            output['images'].append(jimg)
            N += 1

    print('total %d'%(N))
    json.dump(output, open(params['output_json'], 'w'))

def fresh_splitflag(params) :
    pkg = json.load(open(params['input_json'], 'r'))
    imgs_trainval = pkg['images'][0:210000]
    imgs_test = pkg['images'][210000:]
    
    for i in range(len(imgs_trainval)):
        if i < 180000:
            imgs_trainval[i]['split'] = 'train'
        else:
            imgs_trainval[i]['split'] = 'val'
    for i in range(len(imgs_test)):
        imgs_test[i]['split'] = 'test'
    print('assign %d images to %s'%(len(imgs_trainval), 'trainval'))
    print('assign %d images to %s'%(len(imgs_test), 'test'))
    output_trainval = {}
    output_trainval['ix_to_word'] = pkg['ix_to_word']
    output_trainval['images'] = imgs_trainval
    json.dump(output_trainval, open('data/dataset/trainval_180K-30K_meta.json', 'w'))
    output_test = {}
    output_test['ix_to_word'] = pkg['ix_to_word']
    output_test['images'] = imgs_test
    json.dump(output_test, open('data/dataset/test_30K_meta.json', 'w'))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_txt', help='input json file to process into hdf5')
    parser.add_argument('--split', help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
    #fresh_splitflag(params)

