#!/bin/bash

python3 AvrisResumeTraining.py \
  --num_users 3 \
  --num_eves 1 \
  --num_envs 1 \
  --N 64 \
  --seed 100 \
  --max_episodes 20 \
  --init_steps 15000 \
  --init_noise 0.05 \
  --fixed_eve \
  --h_dims 512 \
  --init_batch 64 \
  --capacity 5000