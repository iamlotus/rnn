#!/bin/sh

python3 poem.py --cuda_visible_devices=2 --mode=gen --cell_type=lstm --rnn_size=256 --cuda_visible_devices=3


