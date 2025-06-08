# VT1: Brain MRI Analysis
This project is part of my Master’s studies in the **MSE Data Science program** at the **ZHAW Center for Artificial Intelligence**.

The goal of this project is to explore and implement deep learning techniques on brain MRI data, using two datasets: **IXI** and **ADNI**.

## Project Overview
The project is divided into two main tasks:

### 1. **MRI-Based Prediction (IXI Dataset)**
- Built two CNN models:
  - **Sex classification**
  - **Age regression**
- Developed a **custom "preprocessing" pipeline** that:
  - Fixes scan orientation
  - Converts scans to isotropic voxel space
  - Centers volumes on a fixed 3D grid
  - Integrates on-the-fly spatial augmentations (scaling, rotation, translation)
  
  > The preprocessing is designed for **real-time data augmentation** and is applied during data loading rather than offline preprocessing.

### 2. **Hippocampus Segmentation (ADNI Dataset)**
- Implemented a **U-Net** to segment the **left and right hippocampus** from 3D brain MRIs.

## Project Structure
```
src/
├── datasets/ # Custom PyTorch Dataset classes + preprocessing functions
├── training/ # Model definitions, training loops, validation logic
├── utils/ # Helper functions (e.g., plotting, 3D Grad-CAM)
├── main_age.py # Main script for age regression
├── main_sex.py # Main script for sex classification
└── main_unet.py # Main script for hippocampus segmentation
```

## Datasets
- **IXI** dataset: https://brain-development.org/ixi-dataset/
- **ADNI** dataset: https://adni.loni.usc.edu/


## Work in Progress
This repository is still actively evolving.
