#!/bin/sh

pid=`ps -ef |grep tensorboard |grep -v grep | grep -v tensorboard.sh | awk '{print $2}'`
if [ -n "$pid" ]; then
        echo shutdown current tensorboard pid $pid
        kill $pid
fi

nohup tensorboard --port 10086 --logdir=logs/ > logs/tensorboard.out 2>&1 &
echo start tensorboard