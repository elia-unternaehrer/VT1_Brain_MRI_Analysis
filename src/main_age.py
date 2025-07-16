# %%
from training.model import Age3DCNNBaseline, Age3DCNNOneDense, Age3DCNNTwoDense
from training.train_age import train_age
from utils.dataset_splits import create_ixi_dataset_splits
from utils.plotting import plot_true_vs_pred_identity, plot_true_vs_pred_regression


import torch
from torch.utils.data import DataLoader

#%% ---------------------------------------------------------------------------------
## Create dataset 

# dataset config
shared_args = dict(
    img_dir="../data/IXI/reoriented",
    metadata_path="../data/IXI/IXI.xls",
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

#%% ---------------------------------------------------------------------------------
# train the model with full MRI scanns
train_age(
    Age3DCNNBaseline(),
    lr=0.0001,
    weight_decay=0.0001,
    epochs=5,
    train_loader=train_loader,
    test_loader=test_loader,
    save_path_model='../models/age_pred.pth',
    save_path_plots='../logs/',
    use_wandb=True,
    run_name='test plots',
    aug_config=train_transform
)

# %%
