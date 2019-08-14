# coding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import jieba
import hashlib

def json2csv(json_file) :
    csv_file = open('./tmp/%s.csv'%(json_file.strip('/.').split('/')[-1]), 'w')
    dataset = json.load(open(json_file, 'r'))
    for item in dataset :
        csv_file.write(item['image_id'].strip() + '\n')
        assert(len(item['caption']) == 5)
        for cap in item['caption'] :
            cap = cap.strip().split()
            cap = ''.join(cap)
            #csv_file.write(cap.strip() + '\n')
            cap = jieba.cut(cap, cut_all = False)
            csv_file.write(' '.join(cap) + '\n')
    csv_file.close()

def generate_ref(input_json) :
    dataset = json.load(open(input_json, 'r'))
    ref_dataset = {}
    annotations = []
    images = []
    ann_ind = 1
    for item in dataset :
        img_name = item['image_id'].strip().split('.')[0]
        name_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)
        image = {}
        image['file_name'] = img_name
        image['id'] = name_hash
        images.append(image)
        for cap in item['caption'] :
            cap = cap.strip().split()
            cap = ''.join(cap)
            if len(cap) == 0 :
                continue
            cap = ' '.join(jieba.cut(cap, cut_all = False))
            ann = {}
            ann['caption'] = cap
            ann['id'] = ann_ind
            ann_ind += 1
            ann['image_id'] = name_hash
            annotations.append(ann)
    ref_dataset['annotations'] = annotations
    ref_dataset['images'] = images
    info = {}
    info['contributor'] = 'He Zheng'
    info['description'] = 'CaptionEval'
    info['url'] = 'https://github.com/AIChallenger/AI_Challenger.git'
    info['version'] = '1'
    info['year'] = 2017
    ref_dataset['info'] = info
    licenses = {}
    licenses['url'] = 'https://challenger.ai'
    ref_dataset['licenses'] = licenses
    ref_dataset['type'] = 'captions'
    json.dump(ref_dataset, open('dataset/train_ref.json', 'w'))

def generate_ref_kuaishou(input_json) :
    dataset = json.load(open(input_json, 'r'))
    ref_dataset = {}
    annotations = []
    images = []
    ann_ind = 1
    for item in dataset :
        img_name = item['vid'].strip().split('.')[0]
        name_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)
        image = {}
        image['file_name'] = img_name
        image['id'] = name_hash
        images.append(image)
        for cap in item['caption'] :
            cap = cap.strip().split()
            cap = ''.join(cap)
            if len(cap) == 0 :
                continue
            cap = ' '.join(jieba.cut(cap, cut_all = False))
            ann = {}
            ann['caption'] = cap
            ann['id'] = ann_ind
            ann_ind += 1
            ann['image_id'] = name_hash
            annotations.append(ann)
    ref_dataset['annotations'] = annotations
    ref_dataset['images'] = images
    info = {}
    info['contributor'] = 'He Zheng'
    info['description'] = 'CaptionEval'
    info['url'] = 'https://github.com/AIChallenger/AI_Challenger.git'
    info['version'] = '1'
    info['year'] = 2017
    ref_dataset['info'] = info
    licenses = {}
    licenses['url'] = 'https://challenger.ai'
    ref_dataset['licenses'] = licenses
    ref_dataset['type'] = 'captions'
    json.dump(ref_dataset, open('dataset/kuaishou_ref.json', 'w'))



if __name__ == '__main__' :
    #json2csv('./val/val.json')
    #json2csv('./train/train.json')
    #generate_ref('captions/train.json')
    generate_ref_kuaishou('../../../kuaishou/data/kuaishou_raw.json')
