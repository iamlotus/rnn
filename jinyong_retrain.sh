#!/bin/sh
mkdir -p logs/jinyong model/jinyong
rm -rf logs/jinyong/*  model/jinyong/*
./jinyong_stop_train.sh
./jinyong_start_train.sh


