# VT1: Deep Learning for Brain MRI Analysis
This repository contains code for **VT1: Deep Learning for Brain MRI Analysis**, a Master’s project in the **MSE Data Science program** at the **ZHAW Center for Artificial Intelligence**. We explore deep learning techniques on 3D brain MRI data, tackling both predictive modeling and segmentation tasks.

## Project Overview
We developed three core models on two public datasets, with a real-time preprocessing pipeline underpinning all tasks:

1. **Sex Classification (IXI Dataset)** \
Our simple 3D CNN achieved near-perfect accuracy on full-head T1-weighted MRIs, but accuracy fell to ~80% when trained on brain-extracted scans, demonstrating that non-brain features (e.g. facial structures, skull shape) were driving much of the performance.

2. **Age Regression (IXI Dataset)** \
The baseline 3D CNN (full-head scans) yielded an MAE of 7.79 years; brain extraction caused only a modest performance drop, and neither increasing model capacity nor higher input resolution improved upon this baseline.

3. **Hippocampus Segmentation (ADNI Dataset)** \
After exploring normalization layers and loss functions, the best configuration (3D U-Net with BatchNorm, Dice loss and data augmentation) achieved a Dice score of 0.77 for both hippocampi and a 95th-percentile Hausdorff distance (HD95) of 5.15 mm.

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

## References

- **IXI dataset**: https://brain-development.org/ixi-dataset/  
- **ADNI dataset**: https://adni.loni.usc.edu/  
- **HD-BET:**
Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.
Automated brain extraction of multi-sequence MRI using artificial neural
networks. Hum Brain Mapp. 2019; 1–13. https://doi.org/10.1002/hbm.24750
