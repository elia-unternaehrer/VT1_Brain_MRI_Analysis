#%% ---------------------------------------------------------------------------------
from utils.dataset_splits import create_ixi_dataset_splits
from utils.gradcam3d import gradcam_visualization
from training.train_sex import train_sex
from training.model import Sex3DCNN

from torch.utils.data import DataLoader
import torch

#%% ---------------------------------------------------------------------------------

## Create dataset with full MRI scans

# dataset config
shared_args = dict(
    img_dir="../data/IXI/reoriented",
    metadata_path="../data/IXI/IXI.xls",
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

#%% ---------------------------------------------------------------------------------
# train the model with full MRI scanns
train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=5,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/full_mri_sex_model.pth',
    save_path_plots='../logs/',
    use_wandb=True,
    run_name='test plots',
    aug_config=train_transform
)

#%% ---------------------------------------------------------------------------------
# Load the trained model

trained_model = Sex3DCNN()
trained_model.load_state_dict(torch.load("../models/full_mri_sex_model.pth"))


#%% ---------------------------------------------------------------------------------

## Visualize the MRI slices with Grad-CAM
gradcam_visualization(model=trained_model, layer=1, test_loader=test_loader, num_samples_per_class=2)

#%% ---------------------------------------------------------------------------------

## Create dataset with BET MRI scans
shared_args = dict(
    img_dir="../data/reoriented",
    mask_dir="../data/masks",
    metadata_path="../data/IXI.xls",
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='sex',
)


train_transform={
    "rot_range": (-10, 10),
    "translation_range": (-10, 10)
}

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

#%% ---------------------------------------------------------------------------------
# train the model with BET MRI scanns
train_sex(
    Sex3DCNN(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=15,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/bet_mri_sex_model.pth',
    save_path_plots='../logs/',
    use_wandb=True,
    run_name='BET MRI Scans',
    aug_config=train_transform
)

#%% ---------------------------------------------------------------------------------
# Load the trained model
trained_model = Sex3DCNN()
trained_model.load_state_dict(torch.load('../models/bet_mri_sex_model.pth'))

#%% ---------------------------------------------------------------------------------
## Visualize the MRI slices with Grad-CAM
gradcam_visualization(model=trained_model, test_loader=test_loader, num_samples_per_class=2)

# %%
