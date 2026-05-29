#!/bin/bash

# Set the path to save checkpoints
OUTPUT_DIR='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/pretrain_mae_base_patch8_224'
# path to imagenet-1k train set
DATA_PATH='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/datasets/data03/train'
BoardDir='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/log_tensorboard_mae_base_patch8_224'

# batch_size can be adjusted according to the graphics card
#CUDA_VISIBLE_DEVICES=0,1 python
CUDA_VISIBLE_DEVICES=0 python run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --batch_size 6\
        --opt adamw \
		--model pretrain_mae_base_patch8_224 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 710  \
		--input_size 224 \
		--log_dir ${BoardDir} \
        --output_dir ${OUTPUT_DIR}


