from utils.dataset_splits import create_ixi_dataset_splits
from utils.gradcam3d import gradcam_visualization
from training.train_sex import train_sex
from training.model import Sex3DCNN

from torch.utils.data import DataLoader
import torch

data_path = "/scratch/IXI/reoriented"
mask_path = "/scratch/IXI/masks"
metadata_path = "/scratch/IXI/IXI.xls"

# dataset config
shared_args = dict(
    img_dir=data_path,
    metadata_path=metadata_path,
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='sex',
)

train_transform={
    "rot_range": (-5, 5),
    "translation_range": (-7, 7)
}

train_set, test_set = create_ixi_dataset_splits(
    shared_args,
    train_transform=train_transform,
    test_transform=None,
    train_ratio=0.8,
    seed=42,
    plot_distribution=True
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='/scratch/results/full_mri_sex_model.pth',
    save_path_plots='/scratch/results/plots/Full/',
    use_wandb=True,
    run_name='Full MRI Scans',
    aug_config=train_transform
)

############################################################
# BET MRI scans
############################################################

shared_args = dict(
    img_dir=data_path,
    mask_dir=mask_path,
    metadata_path=metadata_path,
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='sex',
)

train_transform={
    "rot_range": (-5, 5),
    "translation_range": (-7, 7)
}

train_set, test_set = create_ixi_dataset_splits(
    shared_args,
    train_transform=train_transform,
    test_transform=None,
    train_ratio=0.8,
    seed=42,
    plot_distribution=False
)

train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='/scratch/results/bet_mri_sex_model.pth',
    save_path_plots='/scratch/results/plots/BET/',
    use_wandb=True,
    run_name='BET MRI Scans',
    aug_config=train_transform
)