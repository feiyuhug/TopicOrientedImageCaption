#!/bin/bash
dataset=coco
image_fold=data/$dataset/images 
input_json=data/$dataset/${dataset}_meta.json
cnn_model=resnet152
batch_size=20

CUDA_VISIBLE_DEVICES=1 python \
    -u prepro_feats.py \
    --dataset $dataset \
    --input_json $input_json --input_label_h5 data/$dataset/${dataset}_label.h5 \
    --img_fold $image_fold --img_size 320 --img_csize 299 \
    --gpu_num 1 --batch_size $batch_size \
    --use_img 1 --use_topic 0 --use_fc 0\
    --cnn_model $cnn_model --cnn_weight data/cnn_models/${cnn_model}.pth \


