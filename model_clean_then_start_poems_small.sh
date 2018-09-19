#!/bin/sh
mkdir -p logs/poems model/poems
rm -rf logs/poems/*  model/poems/*
./model_stop_poems_train.sh
./model_start_poems_train_small.sh


