import os
import json


def clean_feat(feat_path, split):
    dump = json.load(open('../dataset/trainval_meta_v5k.json'))
    imgs = dump['images']
    count = 0
    for img in imgs :
        if img['split'] == split:
            os.remove(os.path.join(feat_path + '_fc', img['image_id']+'.npy'))
            os.remove(os.path.join(feat_path + '_att', img['image_id']+'.npz'))
            count += 1
    print('count: %d'%(count))
if __name__ == "__main__":
    clean_feat('../dataset/train_val_feat448_log_td_newfeat_t500_bs40_gpu4_rnn2048_2_seed100_epoch5_2', 'train')

