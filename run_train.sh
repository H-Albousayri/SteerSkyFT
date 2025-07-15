#!/bin/bash

python3 AvrisTrain.py \
  --num_users 3 \
  --num_eves 1 \
  --N 64 \
  --seed 100 200 300 \
  --max_episodes 300 \
  --init_steps 500 \
  --init_noise 0.45 \
  --h_dims 512 \
  --init_batch 128 \
  --capacity 100000
