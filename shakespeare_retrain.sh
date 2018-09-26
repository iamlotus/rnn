#!/bin/sh
mkdir -p logs/shakespeare model/shakespeare
rm -rf logs/shakespeare/*  model/shakespeare/*
rm -rf data/shakespeare.train data/shakespeare.dict data/shakespeare.validate
./shakespeare_stop_train.sh
./shakespeare_start_train.sh


