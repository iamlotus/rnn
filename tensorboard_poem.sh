#!/bin/sh
if [ -f ".jytensorboardpid" ]; then
    if [ -d /proc/`cat .jytensorboardpid` ]; then
        echo found running tensorboard pid `cat .jytensorboardpid`
    else
        echo [remove dead pid `cat .jytensorboardpid`] \
        && rm .jytensorboardpid \
        && nohup tensorboard --port 10086 --logdir=logs/jinyong >logs/tensorboard.out 2>&1 & echo $! > .jytensorboardpid \
        && echo [tensorboard started]
    fi
else
    nohup tensorboard --port 10086 --logdir=logs/jinyong >logs/tensorboard.out 2>&1 & echo $! > .jytensorboardpid \
    && echo [tensorboard started]
fi