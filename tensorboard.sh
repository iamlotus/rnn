#!/bin/sh
if [ -f ".tensorboardpid" ]; then
    if [ -d /proc/`cat .tensorboardpid` ]; then
        echo found running tensorboard pid `cat .tensorboardpid`
    else
        echo [remove dead pid `cat .tensorboardpid`] \
        && rm .tensorboardpid \
        && nohup tensorboard --port 10086 --logdir=logs/ >logs/tensorboard.out 2>&1 & echo $! > .tensorboardpid \
        && echo [tensorboard started]
    fi
else
    nohup tensorboard --port 10086 --logdir=logs/ >logs/tensorboard.out 2>&1 & echo $! > .tensorboardpid \
    && echo [tensorboard started]
fi