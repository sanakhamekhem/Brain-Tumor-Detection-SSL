# Brain Tumor Detection SSL

Official implementation of the paper:

**Self-Supervised Vision Transformer for Accurate and Explainable Brain Tumor Detection**

This repository provides the source code, configuration files, and experimental protocol for a self-supervised deep learning framework dedicated to brain tumor detection from medical images. The proposed approach leverages self-supervised representation learning to improve feature extraction, classification performance, and model generalization under limited labeled data conditions.

## Overview

Brain tumor detection from medical images is a challenging task due to visual variability, limited annotated data, and subtle differences between pathological and non-pathological regions. This project investigates the use of self-supervised learning (SSL) with Vision Transformer-based architectures to learn robust visual representations before supervised fine-tuning.

The framework is designed to support:

- Self-supervised pretraining on unlabeled medical images.
- Supervised fine-tuning for brain tumor classification.
- Evaluation using standard classification metrics.
- Explainability analysis to visualize discriminative image regions.
- Reproducible experimentation through configuration files and documented scripts.

## Main Contributions

The main contributions of this repository are:

- A self-supervised learning pipeline for medical image representation learning.
- A Vision Transformer-based classification framework for brain tumor detection.
- A reproducible training and evaluation protocol.
- Explainability support for visual interpretation of model predictions.
- Experimental scripts for comparing SSL-based training with conventional supervised learning.

## Dataset Organization

If a multi-class tumor classification setting is used, the dataset can be organized as follows:

```text
dataset/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

For self-supervised pretraining, class labels are not required. The unlabeled dataset can be organized as follows:

```text
dataset_ssl/
└── unlabeled/
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ...
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Brain-Tumor-Detection-SSL.git
cd Brain-Tumor-Detection-SSL
```

Create and activate a virtual environment.

For Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Self-Supervised Pretraining
python run_mae_pretraining.py \
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
## Supervised Fine-Tuning
python run_class_finetuning.py \
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

## License
This project is released under the MIT License. See the LICENSE file for more details.
