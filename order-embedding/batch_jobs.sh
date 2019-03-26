#########################################################################
# File Name: batch_jobs.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Sat Apr  8 19:53:09 2017
#########################################################################
#!/bin/bash


THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32 python driver_tmp.py order tmp2 2>&1 | tee log_tmp2.txt




