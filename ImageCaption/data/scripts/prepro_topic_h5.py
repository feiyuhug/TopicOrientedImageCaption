import h5py
import numpy as np
import cPickle

def gen_topic_h5_gt(image_ids_file, splits, input_npys):
    image_ids = {}
    with open(image_ids_file) as f:
        indx = 0
        for line in f:
            image_ids[int(line.strip())] = indx
            indx += 1

    split_ids = []
    for split in splits:
        with open(split) as f:
            split_id = []
            for line in f:
                split_id.append(int(line.strip()))
            split_ids.append(split_id)
    
    output = np.zeros((123287, 5), dtype='int32')
    for i, input_npy in enumerate(input_npys):
        topic_npy = np.load(input_npy)
        for j, sid in enumerate(split_ids[i]):
            output[image_ids[sid]] = topic_npy[j*5: (j+1)*5].argmax(axis=1)
    #np.save(open('dump.npy', 'w'), output)
    topic_h5 = h5py.File('dump.h5', 'w')
    topic_h5.create_dataset('topics', dtype='int32', data=output)
    topic_h5.close()



if __name__ == "__main__":    
    gen_topic_h5_gt('data/coco/splits/image_id.txt', \
            ['data/coco/splits/kar_train.ids', \
            'data/coco/splits/kar_test.ids', \
            'data/coco/splits/kar_val.ids'], \
            ['data/coco/topic_model/coco_new/doc-topic_train_line_t200.npy', \
            'data/coco/topic_model/coco_new/doc-topic_test_line_t200.npy', \
            'data/coco/topic_model/coco_new/doc-topic_val_line_t200.npy'])



