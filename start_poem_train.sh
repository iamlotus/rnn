#!/bin/sh

# multiple command in one line


if [ -f ".poemtrainpid" ]; then
    if [ -d /proc/`cat .poemtrainpid` ]; then
        echo found running pid `cat .poemtrainpid`
    else
        rm .poemtrainpid \
        && echo [remove dead pid `cat .poemtrainpid`] \
        && nohup python3 poem.py --mode=train --cell_type=lstm --rnn_size=256 --learning_rate=0.00001 --batch_size=4096 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=20000 --training_echo_interval=1 --validate_echo_interval=200 --training_save_interval=100 >logs/poem/train.out 2>&1 & echo $! > .poemtrainpid \
        && echo [train started] \
        && busybox tail -f logs/poem/train.out
    fi
else
    nohup python3 poem.py --mode=train --cell_type=lstm --rnn_size=256 --learning_rate=0.00001 --batch_size=4096 --train_file_path=./data/poems.txt --validate_file_path=none --epochs=20000 --training_echo_interval=1 --validate_echo_interval=200 --training_save_interval=100 >logs/poem/train.out 2>&1 & echo $! > .poemtrainpid \
    && echo [train started] \
    && busybox tail -f logs/poem/train.out
fi



