#!/bin/sh

# multiple command in one line


if [ -f ".poemtrainpid" ]; then
    if [ -d /proc/`cat .poemtrainpid` ]; then
        echo found running pid `cat .poemtrainpid`
    else
        rm .poemtrainpid \
        && echo [remove dead pid `cat .poemtrainpid`] \
        && nohup python3 poem.py --cuda_visible_devices=0 --mode=train --cell_type=rnn --rnn_size=256 --learning_rate=0.00001 --batch_size=2 --train_file_path=./data/small_poems.txt --validate_file_path=none --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/poem/train.out 2>&1 & echo $! > .poemtrainpid \
        && echo [train started] \
        && busybox tail -f logs/poem/train.out
    fi
else
    nohup python3 poem.py --cuda_visible_devices=0 --mode=train --cell_type=rnn --rnn_size=256 --learning_rate=0.00001 --batch_size=2 --train_file_path=./data/small_poems.txt --validate_file_path=none --epochs=20000 --training_echo_interval=100 --training_save_interval=1000 >logs/poem/train.out 2>&1 & echo $! > .poemtrainpid \
    && echo [train started] \
    && busybox tail -f logs/poem/train.out
fi



