#!/bin/sh

# multiple command in one line
if [ -f ".mjytrainpid" ]; then
    if [ -d /proc/`cat .mjytrainpid` ]; then
        echo found running pid `cat .mjytrainpid`
    else
        rm .mjytrainpid \
        && echo [remove dead pid `cat .mjytrainpid`] \
        && nohup python3 model.py --cuda_visible_devices=1 --mode=train --data_type=jinyong --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/jinyong/train.out 2>&1 & echo $! > .mjytrainpid \
        && echo [train started] \
        && busybox tail -f logs/jinyong/train.out
    fi
else
    nohup python3 model.py --cuda_visible_devices=1 --mode=train --data_type=jinyong --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/jinyong/train.out 2>&1 & echo $! > .mjytrainpid \
    && echo [train started] \
    && busybox tail -f logs/jinyong/train.out
fi

