#!/bin/sh
python3 rnn_tf.py --mode=gen --gpu=1 --input_path=data/jinyong.txt --cell_type=lstm --rnn_size=256 --num_layers=3