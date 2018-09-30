#!/bin/sh

# multiple command in one line
if [ -f ".shakespearepid" ]; then
    if [ -d /proc/`cat .shakespearepid` ]; then
        echo found running pid `cat .shakespearepid`
    else
        rm .shakespearepid \
        && echo [remove dead pid `cat .shakespearepid`] \
        && nohup python3 rnn_tf.py --mode=train --gpu=0 --encoding=utf --learning_rate=0.002 --learning_rate_decay_ratio=0.97 --learning_rate_decay_every=10 --input_path=data/shakespeare.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=500 --print_validate_every=2000 --save_model_every=5000 >logs/shakespeare.out 2>&1 & echo $! > .shakespearepid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
    fi
else
    nohup python3 rnn_tf.py --mode=train --gpu=0 --encoding=utf --learning_rate=0.002 --learning_rate_decay_ratio=0.97 --learning_rate_decay_every=10 --input_path=data/shakespeare.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=500 --print_validate_every=2000 --save_model_every=2000 >logs/shakespeare.out 2>&1 & echo $! > .shakespearepid \
        && echo [shakespeare train started] \
        && busybox tail -f logs/shakespeare.out
fi

