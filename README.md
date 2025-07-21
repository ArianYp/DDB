# DDB: Diffusion Driven Balancing

**Diffusion Driven Balancing (DDB)** is a technique to address spurious correlations in deep neural networks, particularly for image classification tasks. Standard Empirical Risk Minimization (ERM) models often rely on irrelevant features, leading to poor generalization on out-of-distribution data. DDB leverages text-to-image diffusion models to generate balanced training samples, reducing the model's dependence on spurious correlations. accepted at ICCV 2025 main conference.

## Abstract

Deep neural networks trained with Empirical Risk Minimization (ERM) perform well when both training and test data come from the same domain, but they often fail to generalize to out-of-distribution samples. In image classification, these models may rely on spurious correlations that often exist between labels and irrelevant features of images, making predictions unreliable when those features do not exist. We propose a Diffusion Driven Balancing (DDB) technique to generate training samples with text-to-image diffusion models for addressing the spurious correlation problem. First, we compute the best describing token for the visual features pertaining to the causal components of samples by a textual inversion mechanism. Then, leveraging a language segmentation method and a diffusion model, we generate new samples by combining the causal component with the elements from other classes. We also meticulously prune the generated samples based on the prediction probabilities and attribution scores of the ERM model to ensure their correct composition for our objective. Finally, we retrain the ERM model on our augmented dataset. This process reduces the model’s reliance on spurious correlations by learning from carefully crafted samples for in which this correlation does not exist. Our experiments show that across different benchmarks, our technique achieves better worst-group accuracy than the existing state-of-the-art methods.

## Features

- Identifies and isolates causal components in images using textual inversion.
- Generates new training samples with diffusion models by recombining causal and non-causal elements.
- Prunes generated samples using model prediction probabilities and attribution scores.
- Retrains ERM models on the augmented dataset for improved generalization.

## Results

DDB achieves superior worst-group accuracy across multiple benchmarks compared to existing state-of-the-art methods.

## Citation


## Textual Inversion

The textual inversion component is adapted from [Hugging Face Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py). This script allows you to learn a new token representing a specific visual concept (e.g., a dog) using your dataset.

### Usage

To train a textual inversion embedding for a class (e.g., "dog" in the MetaShift dataset), run:

```bash
accelerate launch textual_inversion.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
    --train_data_dir="Textual_inversion_meta/dog" \
    --learnable_property="object" \
    --placeholder_token="<dog>" \
    --initializer_token="animal" \
    --class_name="animal" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=3000 \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="textual_inversion_dog"
```
This will produce a learned embedding for the placeholder token (e.g., `<dog>`), which can be used in subsequent diffusion-based sample generation steps.

## Language Segmentation Model

Our project expects the `languagesegmentanything` folder containing the language segmentation model. You can obtain it by cloning the repository from [This Github repo](https://github.com/luca-medeiros/lang-segment-anything)
