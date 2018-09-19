#!/bin/sh

python3 model.py --cuda_visible_devices=4 --mode=gen --data_type=poems --cell_type=lstm --rnn_size=256 --num_layers=3

