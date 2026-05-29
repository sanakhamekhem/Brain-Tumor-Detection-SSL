#!/bin/bash

MODEL_PATH_TUNED='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/resultatbase2448patch8/checkpoint-best.pth'
OUTPUT_DIR_TEST='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/output_result_test1/'
DATA_PATH_TEST='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/datasets/Data2/val/'
DATA_PATH='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/datasets/Data2/'

CUDA_VISIBLE_DEVICES=0 python /mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/run_class_finetuning.py \
        --model vit_base_patch8_224 \
        --eval_data_path ${DATA_PATH_TEST} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH_TUNED} \
        --batch_size 1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --eval \
        --nb_classes 2 \
        --output_dir ${OUTPUT_DIR_TEST}
