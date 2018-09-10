#!/bin/sh

# multiple command in one line


if [ -f ".trainpid" ]; then
    if [ -d /proc/`cat .trainpid` ]; then
        echo found running pid `cat .trainpid`
    else
        rm .trainpid \
        && echo [remove dead pid `cat .trainpid`] \
        && nohup python3 poem.py --mode=train --learning_rate=0.001 --batch_size=64 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=1000 --training_echo_interval=100 --validate_echo_interval=200 --training_save_interval=500 >logs/train.out 2>&1 & echo $! > .trainpid \
        && echo [train started] \
        && busybox tail -f logs/train.out
    fi
else
    nohup python3 poem.py --mode=train --learning_rate=0.001 --batch_size=64 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=1000 --training_echo_interval=100 --validate_echo_interval=200 --training_save_interval=500 >logs/train.out 2>&1 & echo $! > .trainpid \
    && echo [train started] \
    && busybox tail -f logs/train.out
fi



