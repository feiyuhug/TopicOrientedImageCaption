"""
Dataset loading
"""
import numpy
import paths

def load_dataset(name, cnn, load_train=True, fold=0):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    loc = paths.dataset_dir[name]

    splits = []
    if load_train:
        splits = ['train', 'dev']
    else:
        splits = ['dev', 'test']


    dataset = {}

    for split in splits:
        dataset[split] = {}
        caps = []
        splitName = 'val' if (name == 'coco' or name == 'flickr30k') and split == 'dev' else split
        with open('%s/captions/kar_%s_captions.txt' % (loc, splitName), 'rb') as f:
            for line in f:
                caps.append(line.strip())
            dataset[split]['caps'] = caps
        dataset[split]['ims'] = numpy.load('%s/images/%s.npy' % (loc, splitName))
        dataset[split]['cap_tps'] = numpy.load('%s/topics/t200/topic_gt/doc-topic_%s_line_t200.npy' % (loc, splitName))
        dataset[split]['im_tps'] = numpy.load('%s/topics/t200/topic_gt/doc-topic_%s_t200.npy' % (loc, splitName))
        #dataset[split]['im_tps'] = numpy.load('%s/topics/t200/topic_pred/%s.npy' % (loc, splitName))
        
        # norm topic vectors
        dataset[split]['cap_tps'] = (dataset[split]['cap_tps'].T / (dataset[split]['cap_tps'].max(axis=1) + 1e-30)).T
        dataset[split]['im_tps'] = (dataset[split]['im_tps'].T / (dataset[split]['im_tps'].max(axis=1) + 1e-30)).T
        # handle coco specially by only taking 1k or 5k captions/images
        if split in ['dev', 'test'] and fold >= 0:
            dataset[split]['ims'] = dataset[split]['ims'][fold*1000:(fold+1)*1000]
            dataset[split]['im_tps'] = dataset[split]['im_tps'][fold*1000:(fold+1)*1000]
            dataset[split]['caps'] = dataset[split]['caps'][fold*5000:(fold+1)*5000]
            dataset[split]['cap_tps'] = dataset[split]['cap_tps'][fold*5000:(fold+1)*5000]

    return dataset




