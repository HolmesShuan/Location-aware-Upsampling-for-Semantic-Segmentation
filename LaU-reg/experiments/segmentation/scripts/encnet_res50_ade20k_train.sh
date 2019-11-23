#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ade20k \
    --model encnet --dilated --aux --se-loss --offset-loss --batch-size 16\
    --backbone resnet50 --checkname encnet_resnet50_ade20k_train --no-val --offset-weight 0.3 --location-weight 0.1 --up-factor 4 --batch-size-per-gpu 4 --bottleneck-channel 128 --offset-branch-input-channel 512 --category 150 --base-size 520 --crop-size 480 --downsampled-input-size 60 --epochs 180 --lr 0.004

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ade20k \
    --model encnet --dilated --aux --se-loss --offset-loss \
    --backbone resnet50 --resume /home/shuan/LaU-reg/experiments/segmentation/runs/ade20k/encnet/encnet_resnet50_ade20k_train/179_checkpoint.pth.tar --split val --mode testval --up-factor 4 --batch-size-per-gpu 1 --bottleneck-channel 128 --offset-branch-input-channel 512 --category 150 --base-size 520 --crop-size 480 --downsampled-input-size 60 --ms