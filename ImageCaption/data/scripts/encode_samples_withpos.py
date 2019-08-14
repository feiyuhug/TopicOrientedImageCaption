"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

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
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

# for stanford parser
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import Tree
import time

def extract_NP(tree, pre_w_counts):
    res = []
    w_counts = pre_w_counts
    if type(tree) is Tree:
        for i in range(len(tree)):
            w_counts, re = extract_NP(tree[i], w_counts)
            res.extend(re)
        if tree.label() == 'NP' and len(res) == 0:
            res.append([pre_w_counts, w_counts - pre_w_counts])
    else:
        return pre_w_counts+1, []    
    return w_counts, res

def encode_captions(meta_imgs, pred, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(meta_imgs)
  pred_ = {item['image_id']: item['caption'] for item in pred}
  pred = pred_
  for img in meta_imgs:
    img['final_captions'] = [pred[img['id']].strip().split()]
  M = sum(len(img['final_captions']) for img in meta_imgs) # total number of captions
  imgs = meta_imgs

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  entity_start_jx = np.zeros((M, 10), dtype='uint32')
  entity_length = np.zeros((M, 10), dtype='uint32')
  caption_counter = 0
  counter = 1
  eng_parser = CoreNLPParser(url='http://localhost:9000', encoding='utf8')
  s_time = time.time()
  for i,img in enumerate(imgs):
    if i % 100 == 0:
      print('processing %d/%d, time: %.2f'%(i, len(imgs), (time.time() - s_time)/60))
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'
    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      s_tree = list(eng_parser.parse(s))[0]
      if i % 100 == 0:
        print(s)
        print(s_tree)
      _, marks = extract_NP(s_tree, 0)
      marks_ = []
      for item in marks:
        if item[0] + item[1] <= params['max_length']:
          marks_.append(item)
      marks = marks_
      if len(marks) > 0:
        entity_start_jx[caption_counter][0:len(marks)] = np.array(marks)[:,0]
        entity_length[caption_counter][0:len(marks)] = np.array(marks)[:,1]
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length, entity_start_jx, entity_length

def main(params):

  preds = json.load(open(params['input_json'], 'r'))
  meta = json.load(open(params['input_meta'], 'r'))
  itow = meta['ix_to_word']
  wtoi = {w:i for i,w in itow.iteritems()}
  meta_imgs = meta['images']
  meta_imgs_ = []
  for img in meta_imgs:
    if img['split'] == params['split']:
      meta_imgs_.append(img)
  meta_imgs = meta_imgs_
  seed(123) # make reproducible
  
  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = meta_imgs
  json.dump(out, open(params['output_meta'], 'w'))
  print('wrote ', params['output_meta'])

  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length, entity_start_jx, entity_length = encode_captions(meta_imgs, preds, params, wtoi)

  # create output h5 file
  f_lb = h5py.File(params['output_h5'], "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.create_dataset("entity_start_jx", dtype='uint32', data=entity_start_jx)
  f_lb.create_dataset("entity_length", dtype='uint32', data=entity_length)
  f_lb.close()
  

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_meta', required=True, help='input json file to process into hdf5')
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--split', required=True, help='input json file to process into hdf5')
  
  parser.add_argument('--output_h5', default='data', help='output h5 file')
  parser.add_argument('--output_meta', required=True, help='input json file to process into hdf5')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
