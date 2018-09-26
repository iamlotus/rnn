#!/bin/sh
python3 rnn_tf.py --mode=gen --gpu=4 --encoding=utf --input_path=data/shakespeare.txt  --cell_type=gru --rnn_size=128 --num_layers=2