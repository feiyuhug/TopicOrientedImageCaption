#!/bin/bash

sess=tmp2

THEANO_FLAGS=device=cuda0,floatX=float32,dnn.enabled=False python driver.py order $sess 2>&1 | tee log_$sess.txt





