from datasets.ixi_dataset import IXIDataset
import torch
import utils.plotting as plotting
from torch.utils.data import Subset

def create_ixi_dataset_splits(shared_args, train_transform=None, test_transform=None, train_ratio=0.8, seed=42, plot_distribution=False):

    # Build full dataset just to get stable indices
    full_dataset = IXIDataset(
        transform=None,
        **shared_args
    )

    # Split indices reproducibly
    total_len = len(full_dataset)
    train_size = int(train_ratio * total_len)
    test_size = total_len - train_size
    generator = torch.Generator().manual_seed(seed)

    train_indices, test_indices = torch.utils.data.random_split(
        range(total_len), [train_size, test_size], generator=generator
    )

    # Now make two datasets: one with transforms, one without
    train_dataset = IXIDataset(
        transform=train_transform,
        **shared_args
    )

    test_dataset = IXIDataset(
        transform=test_transform,
        **shared_args
    )

    # Apply the same split indices to both datasets
    train_set = Subset(train_dataset, train_indices)
    test_set = Subset(test_dataset, test_indices)

    print(len(train_set), len(test_set))

    if plot_distribution:
        # plot class distribution
        plotting.plot_sex_distribution(full_dataset, title="Full Dataset Class Distribution")
        plotting.plot_sex_distribution(train_set, title="Train Set Class Distribution")
        plotting.plot_sex_distribution(test_set, title="Test Set Class Distribution")

    return train_set, test_set