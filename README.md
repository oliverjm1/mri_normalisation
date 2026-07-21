# A Systematic Benchmark of Intensity Normalisation Methods for 3D Knee MRI Segmentation and Cross-Domain Generalisability

This repository contains the code and experiments accompanying the paper:

> **A Systematic Benchmark of Intensity Normalisation Methods for 3D Knee MRI Segmentation and Cross-Domain Generalisability**

Accepted for presentation at the **Medical Image Understanding and Analysis (MIUA) 2026** conference.

## Overview

Intensity normalisation is a common preprocessing step in MRI segmentation pipelines, yet its impact on model performance and robustness across imaging domains remains unclear.

This project benchmarks a range of MRI intensity normalisation methods using the nnU-Net framework. Models were trained on the IWOAI 2019 knee MRI dataset and evaluated on both the internal test set and the external SKM-TEA dataset to investigate how different normalisation strategies affect segmentation performance and cross-domain generalisability.

## Data

Training and internal evaluation were performed using the **IWOAI 2019** knee MRI segmentation dataset.

To assess cross-domain generalisability, all trained models were additionally evaluated on the **SKM-TEA** dataset without retraining, allowing the effects of domain shift to be investigated across differing acquisition protocols and imaging characteristics.

The **IWOAI 2019** dataset is available upon request from the challenge repository:
https://github.com/denizlab/2019_IWOAI_Challenge

The **SKM-TEA** dataset is openly available via the Stanford Digital Repository:
https://doi.org/10.71718/2ghb-nv62

Additional information, documentation, and tutorials for SKM-TEA are available at:
https://github.com/StanfordMIMI/skm-tea

## Methods

Seven intensity normalisation methods were compared, which were integrated into the nnU-Net framework:

1. **Z-score** - standardising images by their mean and standard deviation.
1. **Min-max** - scaling intensity values to [0, 1] range.
1. **Robust min-max** - intensity values are clipped to 1st and 99th percentile before [0, 1] scaling.
1. **Histogram Equalisation (HE)** - global histogram equalisation across each image.
1. **Contrast-limited adaptive histogram equalisation (CLAHE)** - image histograms equalised locally using kernels.
1. **Nyúl histogram standardisation** - a template is created from intensity landmarks from the training data. Data is then transformed to match this template.
1. **GMM normalisation** - a four-component gaussian mixture model is fitted to the image intensity distribution, with the second highest peak used to standardise the image.

## Set-up

nnU-Net v2 was installed locally to allow for editing and integration of other normalisation methods. Instructs for how to clone locally can be found [HERE](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/getting-started/installation-and-setup.md).

For code used for normalisation schemes, and information about how these were added to the nnU-Net framework, see `nnUNet/normalisation_functions.py` and `nnUNet/nnUNet.md`.

Creation of dataset folders should follow the instructions on the nnUNetv2 github page, where an `nnUNet_data` folder was created, with `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` subfolders.

Nyúl standardisation landmarks were created and saved in the `nnUNet_preprocessed/{DATASET_NAME}` folder.

## Repository Contents

- Training pipelines
- MRI intensity normalisation methods
- Evaluation scripts
- Statistical analysis
- Figures and results used in the manuscript