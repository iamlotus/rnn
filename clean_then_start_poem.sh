#!/bin/sh
mkdir -p logs/poem model/poem
rm -rf logs/poem/* model/poem/*
./stop_poem_train.sh
./start_poem_train.sh


