#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --dataset pcontext \
    --model encnet --dilated --lateral --aux --se-loss --offset-loss --batch-size 16\
    --backbone resnet101 --checkname encnet_resnet101_pcontext --no-val --offset-weight 0.3 --location-weight 0.1 --up-factor 4 --batch-size-per-gpu 2 --bottleneck-channel 64 --offset-branch-input-channel 512 --category 59 --base-size 520 --crop-size 480 --downsampled-input-size 60
# val [single]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --dataset pcontext \
    --model encnet --dilated --lateral --aux --se-loss --offset-loss \
    --backbone resnet101 --resume /home/shuan/LaU-reg/experiments/segmentation/runs/encnet_resnet101_pcontext/encnet/encnet_resnet101_pcontext/179_checkpoint.pth.tar --split val --mode test --up-factor 4 --batch-size-per-gpu 1 --bottleneck-channel 64 --offset-branch-input-channel 512 --category 59 --base-size 520 --crop-size 480 --downsampled-input-size 60 # --ms  
