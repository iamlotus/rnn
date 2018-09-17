#!/bin/sh
mkdir -p logs/jinyong model/jinyong
rm -rf logs/jinyong/*  model/jinyong/*
./stop_jinyong_train.sh
./start_jinyong_train_small.sh


