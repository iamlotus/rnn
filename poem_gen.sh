#!/bin/sh

python3 poem.py --mode=gen --cuda_visible_devices=1 --batch_size=128 --file_path=./data/poems.txt


