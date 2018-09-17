#!/bin/sh

# multiple command in one line
if [ -f ".jytrainpid" ]; then
    if [ -d /proc/`cat .jytrainpid` ]; then
        echo found running pid `cat .jytrainpid`
    else
        rm .jytrainpid \
        && echo [remove dead pid `cat .jytrainpid`] \
        && nohup python3 jinyong.py --cuda_visible_devices=1 --mode=train --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --save_checkpoints_steps=1000 >logs/jinyong/train.out 2>&1 & echo $! > .jytrainpid \
        && echo [train started] \
        && busybox tail -f logs/jinyong/train.out
    fi
else
    nohup python3 jinyong.py --cuda_visible_devices=1 --mode=train --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --save_checkpoints_steps=1000 >logs/jinyong/train.out 2>&1 & echo $! > .jytrainpid \
    && echo [train started] \
    && busybox tail -f logs/jinyong/train.out
fi



