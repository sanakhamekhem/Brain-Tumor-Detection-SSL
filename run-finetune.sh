# Set the path to save checkpoints
OUTPUT_DIR='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/resultatdata3patch8'
# path to imagenet-1k set
DATA_PATH='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/datasets/Data3/'
# path to pretrain model
MODEL_PATH='/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/pretrain_mae_base_patch8_224/checkpoint-709.pth'

# batch_size can be adjusted according to the graphics card
#CUDA_VISIBLE_DEVICES=0,1 python
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
        --model vit_base_patch8_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 150 \
        --nb_classes 4 \
        --dist_eval

        