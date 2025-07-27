#!/bin/bash

python3 AvrisTrainBL.py \
  --num_users 5 \
  --num_eves 2 \
  --num_envs 5 \
  --M 16 \
  --N 16 \
  --seed 100 200 300 \
  --max_episodes 300 \
  --warmup_episodes 100 \
  --init_steps 500 \
  --init_noise 0.45 \
  --PL_ratio 2.0 \
  --UE_spacing 20 \
  --UAV_height 50 \
  --los \
  --h_dims 256 \
  --init_batch 128 \
  --capacity 100000