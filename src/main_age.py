# %%
import datasets.ixi_dataset as ds
import training.model as model
import training.train_age as tv
import utils.plotting as plotting

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

def reload_imports():
    import importlib
    import datasets.ixi_dataset
    import training.model as model
    import training.train_age as tv
    import utils.plotting as plotting

    importlib.reload(datasets.ixi_dataset)
    importlib.reload(model)
    importlib.reload(tv)
    importlib.reload(plotting)

#%% ---------------------------------------------------------------------------------
reload_imports()
# Common dataset config
shared_args = dict(
    img_dir="/raid/persistent_scratch/untereli/data/img",
    metadata_path="/raid/persistent_scratch/untereli/data/IXI.xls",
    voxel_space=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    label_type='age',
)

# Build full dataset just to get stable indices
full_dataset = ds.IXIDataset(transform=None, **shared_args)

# Split indices reproducibly
total_len = len(full_dataset)
train_size = int(0.8 * total_len)
test_size = total_len - train_size
generator = torch.Generator().manual_seed(42)
train_indices, test_indices = torch.utils.data.random_split(
    range(total_len), [train_size, test_size], generator=generator
)

# Now make two datasets: one with transforms, one without
train_dataset = ds.IXIDataset(
    transform={
        "rot_range": (-10, 10),
        "zoom_range": (0.9, 1.1),
        "translation_range": (-5, 5)
    },
    **shared_args
)

test_dataset = ds.IXIDataset(transform=None, **shared_args)

# Apply the same split indices to both datasets
train_set = Subset(train_dataset, train_indices)
test_set = Subset(test_dataset, test_indices)

print(len(train_set), len(test_set))

# create dataloaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

#%% ---------------------------------------------------------------------------------

ages = [full_dataset[i][1].item() for i in range(len(full_dataset))]
print(f"Min age: {min(ages)}, Max age: {max(ages)}, Mean age: {np.mean(ages):.2f}")

#%% ---------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = model.HigherRes3DCNN().to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

epochs = 10

# Keep track of stats to plot them
train_losses = []
train_r2 = []
val_losses = []
val_r2 = []

for epoch in range(1, epochs + 1):
    train_result = tv.train(epoch, cnn_model, optimizer, device, criterion, train_loader)
    train_losses.append(train_result["loss"])
    train_r2.append(train_result["r2"])
    
    val_result = tv.validate(cnn_model, device, test_loader, criterion)
    val_losses.append(val_result["loss"])
    val_r2.append(val_result["r2"])

    print(val_result['predictions'][:5].squeeze().numpy())
    print(val_result['targets'][:5].squeeze().numpy())

print(f"Training Loss: {train_losses[-1]:.4f}, R2 Accuracy: {train_r2[-1]:.4f}")
print(f"Test Loss: {val_losses[-1]:.4f}, Test R2: {val_r2[-1]:.4f}")

plotting.plot_results(train_losses, train_r2, val_losses, val_r2)

# %%
plotting.plot_results(train_losses, train_r2, val_losses, val_r2)

#%%

test_result = tv.validate(cnn_model, device, test_loader, criterion)

plt.scatter(test_result['targets'], test_result['predictions'], alpha=0.5)
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs True Age")
plt.grid(True)
plt.show()

# %%
