#!/bin/sh

rm -rf logs/jinyong/* model/jinyong/*
./stop_jinyong_train.sh
./start_jinyong_train.sh


