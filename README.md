# ECG Time Series Classification - Machine Learning Pipeline

Architecture of Machine Learning Systems Alternative Exercise - July 2025

## Overview

This project implements a comprehensive machine learning pipeline for ECG (Electrocardiogram) time series classification. The system classifies ECG signals into four categories: Normal, Atrial Fibrillation (AF), Other, and Noisy. The project explores various deep learning architectures, data augmentation techniques, and data reduction methods to achieve optimal classification performance.

## Project Components

The project addresses four main components:

1. **Dataset Exploration**: Comprehensive analysis of ECG data characteristics and patterns
2. **Modeling and Tuning**: Implementation and optimization of neural network architectures
3. **Data Augmentation**: Advanced techniques to improve model performance through synthetic data generation
4. **Data Reduction**: Efficient data compression and representation methods for resource optimization

## Project Structure

```
AMLS_Exercise/
├── README.md                           # Project documentation
├── Project_report.pdf                  # Detailed technical report
├── dependencies.py                     # Dependency management script
├── utils.py                            # Utility functions for data processing
│
├── models/                           # Model architectures and training utilities
│   ├── __init__.py                    # Package initialization
│   ├── baseline_model.py              # STFT-CNN-RNN baseline model
│   ├── resnet_model.py                # 1D ResNet architecture
│   └── training_utils.py              # Training and evaluation utilities
│
├── 1_dataset_exploration/           # Dataset analysis and exploration
│   └── 1_dataset_exploration.ipynb   # Comprehensive data analysis
│
├── 2_modeling_tuning/               # Model development and optimization
│   └── 2_modeling_tuning.ipynb       # Model training and hyperparameter tuning
│
├── 3_augmentation_features/         # Data augmentation techniques
│   └── 3_augmentation.ipynb          # Feature engineering and augmentation
│
├── 4_data_reduction/                # Data compression and reduction
│   └── 4_data_reduction.ipynb        # Data size optimization techniques
│
├── dataset/                          # Data directory (binary files)
├── base.csv                          # Baseline model predictions
├── augment.csv                       # Augmented model predictions
└── reduced.csv                       # Reduced data model predictions
```

## Experimental Results and Findings

### Summary of Final Tuned Results

| Model Variant | Data Strategy | Val Accuracy | Abs Gain vs Baseline | Rel Gain vs Baseline |
|---------------|--------------|--------------|----------------------|----------------------|
| Baseline STFT-CNN-RNN | Original | 0.711 (71.1%) | — | — |
| 1D ResNet | Original | 0.772 (77.2%) | +0.061 | +8.6% |
| Baseline + Augmentation | Augmented | 0.739 (73.87%) | +0.028 | +3.9% |
| ResNet + Augmentation | Augmented | 0.778 (77.75%) | +0.067 | +9.4% |
| ResNet + Augmentation + 50% Stratified Reduction | 50% (augmented subset) | 0.812 (81.15%) | +0.101 | +14.2% |

### Dataset Size vs Performance (ResNet)

| Dataset Portion (of augmented full) | Samples | Storage (MB) | Val Accuracy |
|-------------------------------------|---------|--------------|--------------|
| 10%  | 1,742 | 23.18  | 0.7764 |
| 25%  | 4,354 | 57.93  | 0.7864 |
| 50%  | 8,708 | 114.90 | 0.8115 |
| 100% | 17,418| 232.00 | 0.7750 |

Peak performance occurs at 50% of the (already augmented) dataset—indicative of reduced overfitting and a sweet spot between data diversity and noise accumulation.

### Key Findings

1. Architecture Dominance: Switching to 1D ResNet yields the largest single baseline uplift (+6.1 pts absolute, +8.6% relative) before any data-centric methods.
2. Augmentation Effects: Augmentation improves early convergence and modestly raises final accuracy (Baseline +2.8 pts; ResNet +0.55 pts). Gains are smaller for ResNet, suggesting capacity already leverages existing variability.
3. Counterintuitive Reduction Gain: A stratified 50% subset (with re-augmentation) outperforms the full augmented set (+3.4 pts over ResNet+Aug and +10.1 pts over raw baseline). This implies the full augmented corpus introduces redundancy / mild label noise; curated reduction regularizes implicitly.
4. Minority Class Improvements: F1 for AF and Noisy classes increases notably in the final ResNet (AF: 0.74→0.77; Noisy: 0.78→0.79) while preserving high Normal precision—indicating better balanced sensitivity.
5. Convergence Efficiency: Final ResNet configurations reach competitive accuracy in substantially fewer epochs than the hybrid baseline (early stopping at 26 vs 47 in initial runs; ~35 in final reduced regime with higher ceiling).
6. LR Scheduling + Early Stopping Synergy: Adaptive LR plus patience-based stopping prevented late-epoch overfitting in ResNet; baseline continued to fit training data with stagnant validation after ~20 epochs.



