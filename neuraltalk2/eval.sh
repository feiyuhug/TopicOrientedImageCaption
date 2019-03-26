#########################################################################
# File Name: train.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Tue Apr 18 20:21:18 2017
#########################################################################
#!/bin/bash

#id=0 baseline
CUDA_VISIBLE_DEVICES=7 th eval.lua -input_h5 coco/cocotalk_kar.h5 -input_json coco/cocotalk_kar.json -language_eval 1 -id 0 -model snapshot/model_id0_stage2.t7 -gpuid 0 2>&1 | tee log_training_id0_stage2_test.txt


#id=1 vgg finetuned by t100 lmdb1
#CUDA_VISIBLE_DEVICES=6 th train.lua -input_h5 coco/cocotalk_kar.h5 -input_json coco/cocotalk_kar.json -cnn_proto model/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model ../multi-label/model_vgg19/lmdb1/vgg19_snapshot_iter_25000.caffemodel -checkpoint_path ./snapshot  -language_eval 1 -id 1_stage2 -finetune_cnn_after 0 -start_from snapshot/model_id1.t7 -gpuid 0 2>&1 | tee log_training_id1_stage2.txt


