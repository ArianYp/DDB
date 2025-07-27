# DDB: Diffusion Driven Balancing

**Diffusion Driven Balancing (DDB)** This repository contains the official implementation of our ICCV 2025 main conference paper.

## Abstract

Deep neural networks trained with Empirical Risk Minimization (ERM) perform well when both training and test data come from the same domain, but they often fail to generalize to out-of-distribution samples. In image classification, these models may rely on spurious correlations that often exist between labels and irrelevant features of images, making predictions unreliable when those features do not exist. We propose a Diffusion Driven Balancing (DDB) technique to generate training samples with text-to-image diffusion models for addressing the spurious correlation problem. First, we compute the best describing token for the visual features pertaining to the causal components of samples by a textual inversion mechanism. Then, leveraging a language segmentation method and a diffusion model, we generate new samples by combining the causal component with the elements from other classes. We also meticulously prune the generated samples based on the prediction probabilities and attribution scores of the ERM model to ensure their correct composition for our objective. Finally, we retrain the ERM model on our augmented dataset. This process reduces the model’s reliance on spurious correlations by learning from carefully crafted samples for in which this correlation does not exist. Our experiments show that across different benchmarks, our technique achieves better worst-group accuracy than the existing state-of-the-art methods.


## Dataset

We follow the dataset implementation approach from the [DaC repository](https://github.com/fhn98/DaC/tree/release). Our code supports three datasets: MetaShift, Waterbirds, and CelebA.
### MetaShift

- **Directory:** `data/metashift`
- Please download the MetaShift dataset into the `data/metashift` directory. The dataset can be downloaded form [here](https://drive.usercontent.google.com/download?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT&authuser=0).

Our code expects the following structure:
 *  `data/metashift/train`
 *  `data/metashift/test`

### Waterbirds

- **Directory:** `data/waterbirds`
- Place the Waterbirds dataset in the `data/waterbirds` directory. The dataset can be downloaded from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

Our code expects the following structure:

 *  `data/waterbirds`
 *  `data/waterbirds/metadata.csv`

### CelebA

- **Directory:** `data/celeba`
- Place the Waterbirds dataset in the `data/celeba` directory. The dataset can be downloaded from [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

Our code expects the following structure:
 *  `data/celeba/img_align_celeba`
 *  `data/celeba/celeba_split.csv`
 *  `data/celeba/list_attr_celeba.csv`

## Textual Inversion

The textual inversion component is adapted from [Hugging Face Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py). This script allows you to learn a new token representing a specific visual concept (e.g., a dog) using your dataset.

### Usage

To train a textual inversion embedding for a class (e.g., "bird" in the Waterbirds dataset), run:

```bash
accelerate launch textual_inversion.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
    --train_data_dir="Textual_inversion_meta/wb" \
    --learnable_property="object" \
    --placeholder_token="<bird>" \
    --initializer_token="bird" \
    --class_name="bird" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=3000 \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="textual_inversion_bird"
```
This will produce a learned embedding for the placeholder token (e.g., `<dog>`), which can be used in subsequent diffusion-based sample generation steps.

## Language Segmentation Model

Our project expects the `languagesegmentanything` folder containing the language segmentation model. You can obtain it by cloning the repository from [This Github repo](https://github.com/luca-medeiros/lang-segment-anything)


## ERM Models

The ERM models used in this project are available at the following link:
[ERM Models - Google Drive](https://drive.google.com/drive/folders/1aVhJRmj4KUAHEjoYBuwNu1r_2g1Db8mG?usp=sharing)

**Usage:**
- Download the ERM models from the link above.
- Place the downloaded files in the `models` folder of the project.


## Sample Generation Pipeline (`generate.py`)

This script runs the DDB pipeline to generate counterfactual training samples using Stable Diffusion and LangSAM.

### Steps

1. Select low-loss samples of a target class from a trained ERM model.
2. Segment the causal region (e.g., hair, bird) using LangSAM.
3. Inpaint new images using a textual inversion token with Stable Diffusion.
4. Prune generated samples using attribution scores and classifier confidence.
5. Save filtered samples and metadata for retraining.

### Example

```bash
python generate.py \
  --dataset waterbirds \
  --model_path models/ERM-wb.model \
  --textual_inversion Textual_inversions/textual_inversion_wb \
  --prompt "a photo of a <waterbird> bird" \
  --mask_prompt "bird" \
  --token "<waterbird>" \
  --label 0 \
  --k 1000 \
  --save_dir generated_data/waterbirds-wb/
```

### Retraining Pipeline (`train.py`)

This script retrains the ERM model on the combined dataset of:
- Original training samples
- Generated counterfactuals (from `generate.py`) for both classes

### Key Features

- Loads a pretrained ResNet50 ERM model
- Merges synthetic and real training samples
- Applies per-environment loss weighting (`gamma1`, `gamma2`)
- Evaluates performance per environment during training and testing

### Example

```bash
python train.py \
  --new_data new_data \
  --dataset celeba \
  --model_path models/ERM-wb.model \
  --gamma1 6 \
  --gamma2 10 \
  --lr 5e-6 \
  --epochs 60 \
  --batch_size 32 \
  --output_dir checkpoints/
```

## Citation
