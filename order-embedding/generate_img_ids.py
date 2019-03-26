import json
import numpy
from collections import defaultdict
import os
import paths


def process_dataset(dataset):
    data_dir = paths.dataset_dir[dataset] + '/'
    #images_dir = paths.images_dir[dataset]
    data = json.load(open(data_dir + 'dataset_%s.json' % dataset, 'r'))

    splits = defaultdict(list)
    for im in data['images']:
        split = im['split']
        if split == 'restval':
            split = 'train'
        img_flag = im['filename'].split('.')[0].split('_')[-1].strip()
        img_id = 0
        for i in range(len(img_flag)) :
            img_id = img_id*10 + int(img_flag[i])
        
        splits[split].append(str(img_id))

    for name, filenames in splits.items():
        print name
        with open(data_dir + 'karpathy_split/' + name + '.ids', 'w') as f :
            for item in filenames :
                f.write(item + '\n')
    
if __name__ == '__main__' :
    process_dataset('flickr30k')













