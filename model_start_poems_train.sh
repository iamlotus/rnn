#!/bin/sh

# multiple command in one line
if [ -f ".mpmtrainid" ]; then
    if [ -d /proc/`cat .mpmtrainid` ]; then
        echo found running pid `cat .mpmtrainid`
    else
        rm .mpmtrainid \
        && echo [remove dead pid `cat .mpmtrainid`] \
        && nohup python3 model.py --cuda_visible_devices=3 --mode=train --data_type=poems --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/poems/train.out 2>&1 & echo $! > .mpmtrainid \
        && echo [train started] \
        && busybox tail -f logs/poems/train.out
    fi
else
    nohup python3 model.py --cuda_visible_devices=3 --mode=train --data_type=poems --cell_type=lstm --rnn_size=256 --num_layers=3 --learning_rate=0.00001 --batch_size=64 --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/poems/train.out 2>&1 & echo $! > .mpmtrainid \
    && echo [train started] \
    && busybox tail -f logs/poems/train.out
fi

