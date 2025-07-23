from utils.dataset_splits import create_ixi_dataset_splits
from utils.gradcam3d import gradcam_visualization
from training.train_sex import train_sex
from training.models import Sex3DCNN

from torch.utils.data import DataLoader
import torch

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
    label_type='sex',
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
    plot_distribution=True
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

############################################################
# Baseline (Full Head) Model                               #
############################################################

train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R1_FullHead.pth',
    save_path_plots='../logs/R1_FullHead/',
    use_wandb=True,
    run_name='R1: Full Head Scans',
    aug_config=train_transform
)

############################################################
# GradCam Visualization                                    #
############################################################

# Load the trained model
trained_model = Sex3DCNN()
trained_model.load_state_dict(torch.load("../models/R1_FullHead.pth"))


## Visualize the MRI slices with Grad-CAM
gradcam_visualization(model=trained_model, layer=4, test_loader=test_loader, num_samples_per_class=2)


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
    label_type='sex',
)

# Data augmentation
train_transform={
    "rot_range": (-10, 10),
    "translation_range": (-10, 10)
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
# Brain Extracted Model                                    #
############################################################

train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/R2_BrainExtracted.pth',
    save_path_plots='../logs/R2_BrainExtracted/',
    use_wandb=True,
    run_name='R2: Brain Extracted Scans',
    aug_config=train_transform
)