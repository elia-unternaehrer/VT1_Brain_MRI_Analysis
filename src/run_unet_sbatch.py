from datasets.adni_dataset import ADNIDataset, create_transform
from training.unet3d import UNet3D, UNet3DBatchNorm, UNet3InstanceNorm
from training.train_segmentation import train_segmentation

import torch
from torch.utils.data import DataLoader

data_path = "/scratch/ADNI"

#------------------------------------------------------------------------------------------------
# Baseline with no batchnorm / data augmentation
#------------------------------------------------------------------------------------------------

# create datasets
train_set = ADNIDataset(
    folder_path=data_path,
    split="train",
    patch_size=(140, 140, 140),
    label_prob=0.95
)

val_set = ADNIDataset(
    folder_path=data_path,
    split="test"
)

# # create dataloaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)

class_weights = torch.tensor([0.1, 0.45, 0.45])


train_segmentation(
    model=UNet3DBatchNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="/scratch/results/unet3d_ce_batchnorm_globalNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="CE_BatchNorm_GlobalNorm",
)