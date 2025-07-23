from training.models import Age3DCNNBaseline, Age3DCNNExtended
from training.train_age import train_age
from utils.dataset_splits import create_ixi_dataset_splits

import torch
from torch.utils.data import DataLoader

# Define paths for data, metadata, and masks
data_path = "/scratch/IXI/reoriented"
metadata_path = "/scratch/IXI/IXI.xls"
mask_path = "/scratch/IXI/masks"

############################################################
# Create Dataset and DataLoaders                           #
############################################################

# dataset config
shared_args = dict(
    img_dir=data_path,
    metadata_path=metadata_path,
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='age',
)

# Data augmentation
train_transform={
    "rot_range": (-5, 5),
    "translation_range": (-7, 7)
}

# Create dataset splits
train_set, test_set = create_ixi_dataset_splits(
    shared_args,
    train_transform=train_transform,
    test_transform=None,
    train_ratio=0.8,
    seed=42,
    plot_distribution=False
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

############################################################
# Baseline Model                                           #
############################################################

train_age(
    Age3DCNNBaseline(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R1_Baseline.pth',
    save_path_plots='../logs/R1_Baseline/',
    use_wandb=True,
    run_name='R1: Baseline',
    aug_config=train_transform
)


############################################################
# Create Dataset and DataLoaders for BET Scans             #
############################################################

# dataset config
shared_args = dict(
    img_dir=data_path,
    mask_dir=mask_path,
    metadata_path=metadata_path,
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='age',
)

# Data augmentation
train_transform={
    "rot_range": (-5, 5),
    "translation_range": (-7, 7)
}

# Create dataset splits
train_set, test_set = create_ixi_dataset_splits(
    shared_args,
    train_transform=train_transform,
    test_transform=None,
    train_ratio=0.8,
    seed=42,
    plot_distribution=False
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)


############################################################
# BET Model with Baseline Architecture                     #
############################################################

train_age(
    Age3DCNNBaseline(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R2_BET_Normal.pth',
    save_path_plots='../logs/R2_BET_Normal/',
    use_wandb=True,
    run_name='R2: BET - Normal',
    aug_config=train_transform
)

############################################################
# BET Model with Extended Regression Head                  #
############################################################

train_age(
    Age3DCNNExtended(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R3_BET_Extended.pth',
    save_path_plots='../logs/R3_BET_Extended/',
    use_wandb=True,
    run_name='R3: BET - Extended',
    aug_config=train_transform
)

############################################################
# Create Dataset and DataLoaders for HighRes BET Scans     #
############################################################

# dataset config
shared_args = dict(
    img_dir=data_path,
    mask_dir=mask_path,
    metadata_path=metadata_path,
    voxel_space=(1.0, 1.0, 1.0),
    target_shape=(256, 256, 256),
    label_type='age',
)

# Data augmentation
train_transform={
    "rot_range": (-5, 5),
    "translation_range": (-7, 7)
}

# Create dataset splits
train_set, test_set = create_ixi_dataset_splits(
    shared_args,
    train_transform=train_transform,
    test_transform=None,
    train_ratio=0.8,
    seed=42,
    plot_distribution=False
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

############################################################
# BET HighRes Model with Baseline Architecture             #
############################################################

train_age(
    Age3DCNNBaseline(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R4_BET_HighRes_Normal.pth',
    save_path_plots='../logs/R4_BET_HighRes_Normal/',
    use_wandb=True,
    run_name='R4: BET - HighRes / Normal',
    aug_config=train_transform
)

############################################################
# BET HighRes Model with Extended Regression Head          #
############################################################

train_age(
    Age3DCNNExtended(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R5_BET_HighRes_Extended.pth',
    save_path_plots='../logs/R5_BET_HighRes_Extended/',
    use_wandb=True,
    run_name='R5: BET - HighRes / Extended',
    aug_config=train_transform
)