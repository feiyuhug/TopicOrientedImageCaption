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
import numpy as np

def load_samples(input_json_pre, file_n, sample_n):
    imgs = {}
    for i in range(file_n):
        print('loading %d th json...'%(i))
        frag = json.load(open(input_json_pre + '%d_all.json'%(i), 'r'))
        imgs_ = frag['imgToEval']
        for k in imgs_.keys():
            img_id, cap_ind = k.strip().split('___')
            img_id = int(img_id)
            if img_id not in imgs:
                imgs[img_id] = [None]*sample_n
            imgs[img_id][int(cap_ind)] = [imgs_[k]['Bleu_1'], imgs_[k]['Bleu_2'], imgs_[k]['Bleu_3'], imgs_[k]['Bleu_4'], imgs_[k]['METEOR'], imgs_[k]['ROUGE_L'], imgs_[k]['CIDEr'], imgs_[k]['caption']]
    return imgs

def main(params):
    meta = json.load(open(params['meta_json']))
    imgs_meta = meta['images']
    print('we have %d imgs in meta'%(len(imgs_meta)))
    ix2w = meta['ix_to_word']
    w2ix = {w:ix for ix, w in ix2w.iteritems()}
    imgs = load_samples(params['input_json_pre'], params['input_json_num'], params['sample_n'])
    print('%d samples loaded ...'%(len(imgs.keys())))
    caps = np.zeros((len(imgs_meta), params['sample_n'], params['sample_len']), dtype='uint32')
    cap_scores = np.zeros((len(imgs_meta), params['sample_n'], 7), dtype='float32')
    for k, img_meta in enumerate(imgs_meta):
        img_id = img_meta['id']
        assert img_id in imgs
        for c in range(params['sample_n']):
            b1, b2, b3, b4, M, R, C, sent = imgs[img_id][c]
            cap_scores[k,c,:] = [b1, b2, b3, b4, M, R, C]
            tokens = sent.strip().split()
            assert(len(tokens)) <= params['sample_len']
            for t, tok in enumerate(tokens):
                caps[k, c, t] = w2ix[tok]
    #np.save(open('caps_tmp.npy', 'w'), caps)
    #np.save(open('cap_scores_tmp.npy', 'w'), cap_scores)
    cap_f = h5py.File(params['output_h5'], 'w')
    cap_f.create_dataset("caps", dtype='int32', data=caps)
    cap_f.create_dataset("cap_scores", dtype='float32', data=cap_scores)
    cap_f.close()
    print('wrote to %s'%(params['output_h5']))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # input json files
    parser.add_argument('--meta_json', required=True, help='meta data for training')
    parser.add_argument('--input_json_pre', required=True, help='prefix for input json files ')
    parser.add_argument('--input_json_num', type=int, required=True, help='input json file number')
    
    # param for samples
    parser.add_argument('--sample_n', type=int, required=True, help='sample num for each image')
    parser.add_argument('--sample_len', type=int, required=True, help='sample length for each image')

    # output 
    parser.add_argument('--output_h5', required=True, help='h5file for caption encoding and scores')
    
    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)


