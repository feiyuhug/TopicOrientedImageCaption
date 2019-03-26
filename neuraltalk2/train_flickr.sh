#!/bin/bash

# test
CUDA_VISIBLE_DEVICES=2 th train.lua -net_type 1 -input_h5 ../data/flickr30k/flickr30ktalk_kar.h5 -input_h5_topics ../data/flickr30k/flickr30ktalk_topic_kar.h5 -input_json ../data/flickr30k/flickr30ktalk_kar.json -cnn_proto model/VGG_ILSVRC_19_layers_deploy_emb.prototxt -cnn_model model/VGG_ILSVRC_19_layers_deploy_emb_t100_flickr.caffemodel -fcn_proto model/topic_emb_t100.prototxt -fcn_model model/topic_emb_t100_flickr.caffemodel -checkpoint_path ./snapshot -language_eval 1 -val_images_use 1000 -topic_num 100 -start_from snapshot/model_idmodel2_stage2.t7 -beam_size 3 -id model1_pred -gpuid 0 2>&1 | tee log_pred_beam3.txt

# train
#CUDA_VISIBLE_DEVICES=4 th train.lua -net_type 1 -rnn_size 1024 -input_h5 ../data/flickr30k/flickr30ktalk_kar.h5 -input_h5_topics ../data/flickr30k/flickr30ktalk_topic_kar.h5 -input_json ../data/flickr30k/flickr30ktalk_kar.json -cnn_proto model/VGG_ILSVRC_19_layers_deploy_emb.prototxt -cnn_model model/VGG_ILSVRC_19_layers_deploy_emb_t100_flickr.caffemodel -fcn_proto model/topic_emb_t100.prototxt -fcn_model model/topic_emb_t100_flickr.caffemodel -topic_num 100 -max_iters 100000 -save_checkpoint_every 600 -checkpoint_path ./snapshot  -language_eval 1 -val_images_use 1000 -id model2_stage1 -gpuid 0 2>&1 | tee log/flickr30k/log_model2_stage1.txt

#CUDA_VISIBLE_DEVICES=4 th train.lua -net_type 1 -input_h5 ../data/flickr30k/flickr30ktalk_kar.h5 -input_h5_topics ../data/flickr30k/flickr30ktalk_topic_kar.h5 -input_json ../data/flickr30k/flickr30ktalk_kar.json -cnn_proto model/VGG_ILSVRC_19_layers_deploy_emb.prototxt -cnn_model model/VGG_ILSVRC_19_layers_deploy_emb_t100_flickr.caffemodel -fcn_proto model/topic_emb_t100.prototxt -fcn_model model/topic_emb_t100_flickr.caffemodel -topic_num 100 -max_iters 100000 -save_checkpoint_every 600 -finetune_cnn_after 0 -cnn_learning_rate 1e-5 -start_from snapshot/model_idmodel2_stage1.t7 -learning_rate_decay_start 0 -learning_rate_decay_every 3000 -checkpoint_path ./snapshot  -language_eval 1 -val_images_use 1000 -id model2_stage2 -gpuid 0 2>&1 | tee log/flickr30k/log_model2_stage2.txt

#CUDA_VISIBLE_DEVICES=4 th train.lua -net_type 1 -input_h5 ../data/cocotalk_kar.h5 -input_h5_topics ../data/cocotalk_topic_kar.h5 -input_json ../data/cocotalk_kar.json -cnn_proto model/VGG_ILSVRC_19_layers_deploy_emb.prototxt -cnn_model model/VGG_ILSVRC_19_layers_deploy_emb_t200.caffemodel -fcn_proto model/topic_emb_t200.prototxt -fcn_model model/topic_emb_t200.caffemodel -topic_num 200 -max_iters 20000 -learning_rate 5e-5 -finetune_cnn_after 0 -cnn_learning_rate 1e-6 -finetune_cnn_fcn_w_after 0 -cnn_fcn_w_learning_rate 4e-4 -start_from snapshot/model1_stage2_.t7 -checkpoint_path ./snapshot  -language_eval 1 -id model1_stage3 -gpuid 0 2>&1 | tee log_model1_stage3.txt



