#!/bin/bash
model=show_tell
cnn_model=resnet152
dataset=coco
seq_per_img=5
rnn_size=1024
num_layers=1
dropout=0.5
batch_size=100
gpu_num=1
seed=100
iters=1
id=${model}_${dataset}_dropout${dropout}_bs${batch_size}_iter${iters}_gpu${gpu_num}_rnn${rnn_size}_numlayers${num_layers}_seed${seed}_${cnn_model}
if [ ! -d "log/$id"  ]; then
    mkdir -p log/$id
fi

CUDA_VISIBLE_DEVICES=1 python \
    -u train_super.py --id $id --caption_model ${model} --drop_prob_lm $dropout\
    --dataset $dataset --seq_per_img $seq_per_img\
    --seed $seed --rnn_size $rnn_size  --num_layers $num_layers \
    --input_encoding_size 1024 --topic_num 100 \
    --use_img 0 --use_topic 0 --use_fc 1 \
    --input_json data/$dataset/${dataset}_meta.json \
    --img_fold data/$dataset/images --img_size 320 --img_csize 299 \
    --input_fc_dir data/$dataset/feat299_${cnn_model}_fc\
    --input_label_h5 data/$dataset/${dataset}_label.h5 \
    --batch_size $batch_size --iter_times $iters --gpu_num $gpu_num \
    --fix_rnn 0 --learning_rate 5e-4 --learning_rate_decay_start 10 \
    --learning_rate_decay_every 3 --learning_rate_decay_rate 0.8 \
    --scheduled_sampling_start 10 --scheduled_sampling_increase_every 4 \
    --scheduled_sampling_increase_prob 0.05 --scheduled_sampling_max_prob 0.25 \
    --finetune_cnn_after 10 --cnn_learning_rate 1e-5 \
    --cnn_weight_decay 1e-8 \
    --cnn_model $cnn_model --cnn_weight data/cnn_models/${cnn_model}.pth \
    --checkpoint_path log/$id \
    --save_every 5 \
    --val_images_use 5000 --language_eval 1 --max_epochs 50 \
    2>&1 | tee log/$id/train.log





