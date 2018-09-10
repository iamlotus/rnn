#!/bin/sh

# multiple command in one line


if [ -f ".trainpid" ]; then
    if [ -d /proc/`cat .trainpid` ]; then
        echo found running pid `cat .trainpid`
    else
        rm .trainpid \
        && echo [remove dead pid `cat .trainpid`] \
        && nohup python3 poem.py --mode=train --cell_type=lstm --learning_rate=0.001 --batch_size=16 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=10000 --training_echo_interval=500 --validate_echo_interval=200 --training_save_interval=5000 >logs/train.out 2>&1 & echo $! > .trainpid \
        && echo [train started] \
        && busybox tail -f logs/train.out
    fi
else
    nohup python3 poem.py --mode=train --cell_type=lstm --learning_rate=0.001 --batch_size=16 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=10000 --training_echo_interval=500 --validate_echo_interval=200 --training_save_interval=5000 >logs/train.out 2>&1 & echo $! > .trainpid \
    && echo [train started] \
    && busybox tail -f logs/train.out
fi



