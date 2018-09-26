#!/bin/sh

# multiple command in one line
if [ -f ".jytrainpid" ]; then
    if [ -d /proc/`cat .jytrainpid` ]; then
        echo found running pid `cat .jytrainpid`
    else
        rm .jytrainpid \
        && echo [remove dead pid `cat .jytrainpid`] \
        && nohup python3 rnn_tf.py --mode=train --gpu=0 --input_path=data/jinyong.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=10 --print_validate_every=100 --save_model_every=500 >logs/jinyong.out 2>&1 & echo $! > .jytrainpid \
        && echo [jinyong train started] \
        && busybox tail -f logs/jinyong.out
    fi
else
    nohup python3 rnn_tf.py --mode=train --gpu=0 --input_path=data/jinyong.txt --cell_type=lstm --rnn_size=128 --num_layers=2 --max_epochs=20000 --print_train_every=10 --print_validate_every=100 --save_model_every=500 >logs/jinyong.out 2>&1 & echo $! > .jytrainpid \
    && echo [jinyong train started] \
    && busybox tail -f logs/jinyong.out
fi

