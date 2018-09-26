#!/bin/sh

# multiple command in one line
if [ -f ".sptrainpid" ]; then
    if [ -d /proc/`cat .sptrainpid` ]; then
        echo found running pid `cat .sptrainpid`
    else
        rm .sptrainpid \
        && echo [remove dead pid `cat .sptrainpid`] \
        && nohup python3 rnn_tf.py --mode=train --encoding=utf--gpu=0 --learning_rate=0.01 --input_path=data/shakespeare.txt --cell_type=gru --rnn_size=128 --num_layers=2 --max_epochs=2000 --print_train_every=100 --print_validate_every=500 --save_model_every=2000 >logs/shakespeare.out 2>&1 & echo $! > .sptrainpid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
    fi
else
    nohup python3 rnn_tf.py --mode=train --gpu=0 --encoding=utf --learning_rate=0.01 --input_path=data/shakespeare.txt --cell_type=gru --rnn_size=128 --num_layers=2 --max_epochs=2000 --print_train_every=100 --print_validate_every=500 --save_model_every=2000 >logs/shakespeare.out 2>&1 & echo $! > .sptrainpid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
fi

