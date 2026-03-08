#!/bin/bash

python3 AvrisTrainBL.py \
  --num_users 3 \
  --num_eves 3 \
  --num_envs 5 \
  --M 16 \
  --N 16 \
  --seed 400 500 600 \
  --max_episodes 300 \
  --warmup_episodes 100 \
  --init_steps 500 \
  --init_noise 0.45 \
  --PL_ratio 2.0 \
  --UE_spacing 10 \
  --UAV_height 50 \
  --x_eve_boundry 20 \
  --y_eve_boundry 70 \
  --state_setup Angle \
  --reward_setup rate \
  --los \
  --h_dims 512 \
  --init_batch 128 \
  --last_batch 512 \
  --capacity 100000