#!/bin/sh

# multiple command in one line
if [ -f ".sptrainpid" ]; then
    if [ -d /proc/`cat .sptrainpid` ]; then
        echo found running pid `cat .sptrainpid`
    else
        rm .sptrainpid \
        && echo [remove dead pid `cat .sptrainpid`] \
        && nohup python3 rnn_tf.py --mode=train --gpu=0 --encoding=utf --learning_rate=0.002 --learning_rate_decay_ratio=0.97 --learning_rate_decay_every=10 --input_path=data/shakespeare.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=500 --print_validate_every=5000 --save_model_every=5000 >logs/shakespeare.out 2>&1 & echo $! > .sptrainpid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
    fi
else
    nohup python3 rnn_tf.py --mode=train --gpu=0 --encoding=utf --learning_rate=0.002 --learning_rate_decay_ratio=0.97 --learning_rate_decay_every=10 --input_path=data/shakespeare.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=500 --print_validate_every=5000 --save_model_every=5000 >logs/shakespeare.out 2>&1 & echo $! > .sptrainpid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
fi

