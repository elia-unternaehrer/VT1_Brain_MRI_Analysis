#%% ---------------------------------------------------------------------------------

from datasets.adni_dataset import ADNIDataset, create_transform
from training.unet3d import UNet3D
from training.train_segmentation import train_segmentation

import torch
from torch.utils.data import DataLoader
#%% ---------------------------------------------------------------------------------
data_path = "/raid/persistent_scratch/untereli/ADNI"

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

# create dataloaders
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)


class_weights = torch.tensor([0.1, 0.45, 0.45])
#%% ---------------------------------------------------------------------------------

train_segmentation(
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=3,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="/scratch/results/unet3d_adni.pth",
    use_wandb=True,
    run_name="test_with_sbatch"
)
#%% ---------------------------------------------------------------------------------

import torchio as tio

dataaug_config = {
    "RandomAffine": {
        "scales": (0.9, 1.1),
        "degrees": 10,
        "p": 0.75
    },
    "RandomElasticDeformation": {
        "num_control_points": 7, 
        "max_displacement": 7.5,
        "p": 0.5
    },
    "RandomNoise": {
        "std": (0, 0.01),
        "p": 0.25
    },
    "RandomBiasField": {
        "coefficients": 0.5,
        "p": 0.2
    }
}

transform = create_transform(dataaug_config)

# create datasets
train_set = ADNIDataset(
    folder_path=data_path,
    split="train",
    patch_size=(140, 140, 140),
    label_prob=0.95,
    transform=transform

)
val_set = ADNIDataset(
    folder_path=data_path,
    split="test",
)

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)

class_weights = torch.tensor([0.1, 0.45, 0.45])

train_segmentation(
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    save_path="/scratch/results/unet3d_adni_dataAug.pth",
    use_wandb=True,
    run_name="unet3d_adni_dataAug",
    aug_config=dataaug_config
)
