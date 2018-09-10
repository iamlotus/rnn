#!/bin/sh

python3 poem.py --mode=gen --cuda_visible_devices=1 --batch_size=1 --train_file_path=./data/small_poems.txt


