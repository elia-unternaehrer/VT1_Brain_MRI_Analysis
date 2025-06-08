#%% ---------------------------------------------------------------------------------
from datasets.adni_dataset import ADNIDataset
from training.unet3d import UNet3D
from training.train_segmentation import train_segmentation

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#%% ---------------------------------------------------------------------------------
# create datasets
train_set = ADNIDataset(
    folder_path="../data/ADNI",
    split="train",
    patch_size=(140, 140, 140),
    label_prob=0.95
)
val_set = ADNIDataset(
    folder_path="../data/ADNI",
    split="test",
    patch_size=(140, 140, 140),
    label_prob=0.95
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=16)

# class_count = np.zeros(3)

# for i in range(len(train_set)):
#     _, label = train_set[i]
#     for j in range(3):
#         class_count[j] += torch.sum(label == j).item()

# print(f"Class count: {class_count}")
# class_distribution = class_count / np.sum(class_count)
# class_weights = (1.0 / class_distribution).clip(min=1e-5)
# class_weights = class_weights / np.sum(class_weights)
# print(f"Class distribution: {class_distribution}")
# print(f"Class weights: {class_weights}")

class_weights = torch.tensor([0.1, 0.45, 0.45])


#%% ---------------------------------------------------------------------------------
train_segmentation(
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/unet3d_adni.pth",
)

#%% ---------------------------------------------------------------------------------
import torchio as tio

# Define the transform
transform = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.75), # Random affine transformation (rotation and scaling on 75% of the data)
    tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, p=0.5), # Random elastic deformation on 50% of the data
    tio.RandomNoise(std=(0, 0.1), p=0.25), # Random noise on 25% of the data
    tio.RandomBiasField(coefficients=0.5, p=0.2), # Random bias field on 20% of the data
])


# create datasets
train_set = ADNIDataset(
    folder_path="../data/ADNI",
    split="train",
    patch_size=(140, 140, 140),
    label_prob=0.95,
    transform=transform

)
val_set = ADNIDataset(
    folder_path="../data/ADNI",
    split="test",
    patch_size=(140, 140, 140),
    label_prob=0.95
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)

class_weights = torch.tensor([0.1, 0.45, 0.45])
#%% ---------------------------------------------------------------------------------

train_segmentation(
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/unet3d_adni_dataAug.pth",
)

# %%
