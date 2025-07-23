from datasets.adni_dataset import ADNIDataset, create_transform
from training.models import UNet3D, UNet3DBatchNorm, UNet3DInstanceNorm
from training.train_segmentation import train_segmentation

import torch
from torch.utils.data import DataLoader
import torchio as tio


# Define paths for data
data_path = "/scratch/ADNI"

############################################################
# Create Dataset and DataLoaders                           #
############################################################

# create train and validation datasets
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

# create dataloaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)


class_weights = torch.tensor([0.1, 0.45, 0.45])

############################################################
# Baseline                                                 #
############################################################

train_segmentation(
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/R1_Baseline.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R1: Baseline",
)

############################################################
# R2: Unet with BatchNorm                                  #
############################################################

train_segmentation(
    model=UNet3DBatchNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/R2_CrossEntropy_BatchNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R2: CrossEntropy + BatchNorm",
)

############################################################
# R3: Unet with InstanceNorm                               #
############################################################

train_segmentation(
    model=UNet3DInstanceNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/R3_CrossEntropy_InstanceNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R3: CrossEntropy + InstanceNorm",
)

############################################################
# R3: Unet with InstanceNorm                               #
############################################################

train_segmentation(
    model=UNet3DInstanceNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="../models/R3_CrossEntropy_InstanceNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R3: CrossEntropy + InstanceNorm",
)

############################################################
# R4: Unet with BatchNorm + Dice                           #
############################################################

train_segmentation(
    model=UNet3DBatchNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    loss="dice",
    save_path="../models/R4_Dice_BatchNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R4: Dice + BatchNorm",
)

############################################################
# R5: Unet with BatchNorm + Combined Loss                  #
############################################################

train_segmentation(
    model=UNet3DBatchNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    loss="combined",
    save_path="../models/R5_combined_BatchNorm.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R5: Combined + BatchNorm",
)

############################################################
# Create Dataset and DataLoaders with Data Augmentation    #
############################################################

# Define data augmentation configuration
dataaug_config = {
    "RandomAffine": {
        "scales": (0.9, 1.1),
        "degrees": 5,
        "p": 1
    },
    "RandomElasticDeformation": {
        "num_control_points": 7, 
        "max_displacement": 7,
        "p": 1
    },
}

# Create data augmentation transform
transform = create_transform(dataaug_config)

# create trainset with augmentation
train_set = ADNIDataset(
    folder_path=data_path,
    split="train",
    patch_size=(140, 140, 140),
    label_prob=0.95,
    transform=transform
)

# create validation set without augmentation
val_set = ADNIDataset(
    folder_path=data_path,
    split="test",
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)

class_weights = torch.tensor([0.1, 0.45, 0.45])

############################################################
# R6: Unet with BatchNorm + Dice + Data Augmentation       #
############################################################

train_segmentation(
    model=UNet3DBatchNorm(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    loss="dice",
    save_path="../models/R6_Dice_BatchNorm_DataAug.pth",
    use_hausdorff=False,
    use_wandb=True,
    run_name="R6: Dice + BatchNorm + DataAug",
    aug_config=dataaug_config
)