from datasets.adni_dataset import ADNIDataset, create_transform
from training.unet3d import UNet3D
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
    model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
    train_loader=train_loader,
    test_loader=val_loader,
    epochs=70,
    lr=1e-4,
    weight_decay=1e-5,
    weights=class_weights,
    loss="dice",
    save_path="/scratch/results/unet3d_adni.pth",
    use_hausdorff=True,
    use_wandb=True,
    run_name="Baseline (DICE)",
)

# #------------------------------------------------------------------------------------------------
# # With data augmentation transform and augmentation
# #------------------------------------------------------------------------------------------------
# dataaug_config = {
#     "RandomAffine": {
#         "scales": (0.9, 1.1),
#         "degrees": 5,
#         "p": 1
#     },
#     "RandomElasticDeformation": {
#         "num_control_points": 7, 
#         "max_displacement": 7,
#         "p": 1
#     }
# }


# transform = create_transform(dataaug_config)

# # create trainset with augmentation
# train_set = ADNIDataset(
#     folder_path=data_path,
#     split="train",
#     patch_size=(140, 140, 140),
#     label_prob=0.95,
#     transform=transform
# )

# train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=16)

# train_segmentation(
#     model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
#     train_loader=train_loader,
#     test_loader=val_loader,
#     epochs=70,
#     lr=1e-4,
#     weight_decay=1e-5,
#     weights=class_weights,
#     save_path="/scratch/results/unet3d_adni.pth",
#     use_hausdorff=True,
#     loss="combined",
#     use_wandb=True,
#     run_name="Augmentation (combined loss)",
# )

# train_segmentation(
#     model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
#     train_loader=train_loader,
#     test_loader=val_loader,
#     epochs=70,
#     lr=1e-4,
#     weight_decay=1e-5,
#     weights=class_weights,
#     save_path="/scratch/results/unet3d_adni.pth",
#     use_hausdorff=True,
#     loss="dice",
#     use_wandb=True,
#     run_name="Augmentation (Dice Loss)",
# )

# train_segmentation(
#     model=UNet3D(in_channels=1, out_channels=3, base_filters=32),
#     train_loader=train_loader,
#     test_loader=val_loader,
#     epochs=70,
#     lr=1e-4,
#     weight_decay=1e-5,
#     weights=class_weights,
#     save_path="/scratch/results/unet3d_adni.pth",
#     use_hausdorff=True,
#     use_wandb=True,
#     run_name="Augmentation (CrossEntropy Loss)",
# )