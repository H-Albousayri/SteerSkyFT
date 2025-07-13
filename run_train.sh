#!/bin/bash

python3 AvrisTrain.py \
  --num_users 5 \
  --num_eves 1 \
  --seed 100 200 300 \
  --max_episodes 300 \
  --init_steps 500 \
  --init_noise 0.45 \
  --init_batch 256 \
  --capacity 100000
