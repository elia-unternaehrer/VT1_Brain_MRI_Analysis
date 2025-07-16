from training.model import Age3DCNNBaseline, Age3DCNNOneDense, Age3DCNNTwoDense
from training.train_age import train_age
from utils.dataset_splits import create_ixi_dataset_splits

import torch
from torch.utils.data import DataLoader

data_path = "/scratch/IXI/reoriented"
metadata_path = "/scratch/IXI/IXI.xls"
mask_path = "/scratch/IXI/masks"

# dataset config
shared_args = dict(
    img_dir=data_path,
    mask_dir=mask_path,
    metadata_path=metadata_path,
    voxel_space=(1.0, 1.0, 1.0),
    target_shape=(256, 256, 256),
    label_type='age',
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

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

train_age(
    Age3DCNNBaseline(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='/scratch/results/age_pred_BET_HR_baseline.pth',
    save_path_plots='/scratch/results/plots/BET_HR_Baseline/',
    use_wandb=True,
    run_name='BET / HR - Baseline',
    aug_config=train_transform
)

############################################################
# One Dense Layer Model                                    #
############################################################

train_age(
    Age3DCNNOneDense(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='/scratch/results/age_pred_BET_HR_one_dense.pth',
    save_path_plots='/scratch/results/plots/BET_HR_One_dense/',
    use_wandb=True,
    run_name='BET / HR - One Dense',
    aug_config=train_transform
)

# ############################################################
# # Two Dense Layer Model                                    #
# ############################################################

# train_age(
#     Age3DCNNOneDense(),
#     lr=0.0001,
#     weight_decay=0.0001,
#     epochs=40,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     save_path_model='/scratch/results/age_pred_BET_two_dense.pth',
#     save_path_plots='/scratch/results/plots/BET_Two_dense/',
#     use_wandb=True,
#     run_name='BET - Two Dense',
#     aug_config=train_transform
# )