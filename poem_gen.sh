#!/bin/sh

python3 poem.py --mode=gen --cuda_visible_devices=1 --file_path=./data/poems.txt --epochs=10000 --training_echo_interval=200 --training_save_interval=2000


