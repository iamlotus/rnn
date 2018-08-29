#!/bin/sh

nohup tensorboard --port 10086 --logdir=logs/ > logs/tensorboard.out 2>&1 &



