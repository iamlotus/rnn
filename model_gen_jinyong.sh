#!/bin/sh

python3 model.py --cuda_visible_devices=2 --mode=gen --data_type=jinyong  --cell_type=lstm --rnn_size=256 --num_layers=3

