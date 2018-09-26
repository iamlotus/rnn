#!/bin/sh
mkdir -p logs/shakespare model/shakespare
rm -rf logs/shakespare/*  model/shakespare/*
rm -rf data/shakespare.train data/shakespare.dict data/shakespare.validate
./shakespare_stop_train.sh
./shakespare_start_train.sh


