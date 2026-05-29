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

## Repository Structure

```text
Brain-Tumor-Detection-SSL/
├── configs/
│   └── config.yaml
├── data/
│   └── README.md
├── models/
│   ├── vit_model.py
│   └── ssl_backbone.py
├── scripts/
│   ├── train_ssl.py
│   ├── train_classifier.py
│   ├── evaluate.py
│   └── explain.py
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── visualization.py
├── results/
│   └── README.md
├── requirements.txt
├── LICENSE
└── README.md
