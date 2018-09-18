#!/bin/sh
mkdir -p logs/jinyong model/jinyong
rm -rf logs/jinyong/*  model/jinyong/*
./model_stop_jinyong_train.sh
./model_start_jinyong_train.sh


