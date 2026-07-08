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

## Repository Contents

- Training pipelines
- MRI intensity normalisation methods
- Evaluation scripts
- Statistical analysis
- Figures and results used in the manuscript