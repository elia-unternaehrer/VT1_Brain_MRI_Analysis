import matplotlib.pyplot as plt
from collections import Counter
import torch
import seaborn as sns
import numpy as np

plt.rcParams.update({
    "text.usetex":       False,
    "font.family":       "serif",
    "font.serif":        ["Times New Roman"],
    "figure.figsize":    (5.0, 5.0),
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    15,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.autolayout": True,
})

sns.set_theme(style='white', font_scale=1.2)

def plot_mri_slices(volume_np):
    x, y, z = volume_np.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sagittal: [Z, Y] → transpose for display
    axes[0].imshow(volume_np[x // 2, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Sagittal (X)")

    # Coronal: [Z, X] → transpose for display
    axes[1].imshow(volume_np[:, y // 2, :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal (Y)")

    # Axial: [Y, X] → already displayable
    axes[2].imshow(volume_np[:, :, z // 2].T, cmap="gray", origin="lower")
    axes[2].set_title("Axial (Z)")

    plt.tight_layout()
    plt.show()

def plot_mri_with_cam(mri_tensor, cam_tensor, label=None, pred=None, x_slice=None, y_slice=None, z_slice=None):
    label_map = {0: "Male", 1: "Female"}
    label_text = f"Label: {label_map.get(label, label)}" if label is not None else ""
    pred_text = f" Pred: {label_map.get(pred, pred)}" if pred is not None else ""

    mri = mri_tensor.squeeze().cpu().numpy()
    cam = cam_tensor.squeeze().cpu().numpy()

    x, y, z = mri.shape
    x_slice = x_slice if x_slice is not None else x // 2
    y_slice = y_slice if y_slice is not None else y // 2
    z_slice = z_slice if z_slice is not None else z // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(label_text+pred_text, fontsize=16)

    axes[0].imshow(mri[x_slice, :, :].T, cmap="gray", origin="lower")
    axes[0].imshow(cam[x_slice, :, :].T, cmap="plasma", alpha=0.5, origin="lower")
    axes[0].set_title(f"Sagittal (X={x_slice})")

    axes[1].imshow(mri[:, y_slice, :].T, cmap="gray", origin="lower")
    axes[1].imshow(cam[:, y_slice, :].T, cmap="plasma", alpha=0.5, origin="lower")
    axes[1].set_title(f"Coronal (Y={y_slice})")

    axes[2].imshow(mri[:, :, z_slice].T, cmap="gray", origin="lower")
    axes[2].imshow(cam[:, :, z_slice].T, cmap="plasma", alpha=0.5, origin="lower")
    axes[2].set_title(f"Axial (Z={z_slice})")

    plt.tight_layout()
    plt.show()



def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss', color='orange')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Plot accuracy
    axs[1].plot(train_accuracies, label='Train Accuracy')
    axs[1].plot(val_accuracies, label='Validation Accuracy', color='orange')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot_segmentation_results(train_losses, train_dices, train_avg_dice, val_losses, val_dices, val_avg_dice, val_mean_hd95):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot loss
    axs[0,0].plot(train_losses, label='Train Loss', color='red')
    axs[0,0].plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    axs[0,0].set_title('Loss')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Loss')
    axs[0,0].legend()

    # Plot HD95
    axs[0,1].plot(val_mean_hd95, label='Validation HD95', color='purple')
    axs[0,1].set_title('HD95')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('HD95')
    axs[0,1].legend()

    colors = ['orange', 'blue', 'green']

    # Plot dice
    axs[1,0].plot(train_avg_dice, label='Train Avg Dice', color='red')
    axs[1,0].plot(val_avg_dice, label='Test Avg Dice', color='red', linestyle='--')
    for i in range(len(train_dices[0])):
        axs[1,0].plot([x[i] for x in train_dices], label=f'Train Dice Class {i}', color=colors[i])
        axs[1,0].plot([x[i] for x in val_dices], label=f'Validation Dice Class {i}', color=colors[i], linestyle='--')

    axs[1,0].set_title('Dice Score')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('Accuracy')
    axs[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_sex_distribution(dataset, title="Sex Distribution"):
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))

    metadata = base_dataset.metadata
    subject_ids = base_dataset.subject_ids

    sex_labels = []
    for idx in indices:
        subject_id = subject_ids[idx]
        row = metadata[metadata["IXI_ID"] == subject_id]
        if not row.empty:
            sex_id = int(row.iloc[0]["SEX_ID (1=m, 2=f)"]) - 1
            sex_labels.append(sex_id)

    class_counts = Counter(sex_labels)
    classes = ["Male", "Female"]
    counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    # Plot
    plt.figure(figsize=(5, 4))
    bars = plt.bar(classes, counts, color=["skyblue", "lightpink"])

    # Add counts and percentages above bars
    for bar, count, percent in zip(bars, counts, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{count} ({percent:.1f}%)", ha="center", va="bottom")

    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_true_vs_pred_identity(true_ages, pred_ages, figsize=(6, 6), save_path=None):
    """
    Scatter‐plot True vs. Predicted ages with the identity line y = x.
    
    Parameters
    ----------
    true_ages : array‐like of shape (n_samples,)
        The ground‐truth ages.
    pred_ages : array‐like of shape (n_samples,)
        The ages predicted by the model.
    figsize : tuple of two ints, default=(6, 6)
        Figure size in inches.
    """

    save_path = save_path + "identity.png" if save_path else "identity.png"

    true_ages = np.asarray(true_ages)
    pred_ages = np.asarray(pred_ages)

    # determine plotting range
    lo = min(true_ages.min(), pred_ages.min())
    hi = max(true_ages.max(), pred_ages.max())

    # create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(true_ages, pred_ages, alpha=0.6, edgecolors='k')
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='Ideal (y=x)')

    ax.set_xlabel('True Age (years)')
    ax.set_ylabel('Predicted Age (years)')
    ax.set_title('True vs. Predicted Age (Identity Line)')
    ax.axis('square')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_true_vs_pred_regression(true_ages, pred_ages, figsize=(6, 6), save_path=None):
    """
    Scatter‐plot True vs. Predicted ages with both the identity line and 
    a fitted linear regression line.
    
    Parameters
    ----------
    true_ages : array‐like of shape (n_samples,)
        The ground‐truth ages.
    pred_ages : array‐like of shape (n_samples,)
        The ages predicted by the model.
    figsize : tuple of two ints, default=(6, 6)
        Figure size in inches.
    """
    save_path = save_path + "identity_regression.png" if save_path else "identity_regression.png"

    true_ages = np.asarray(true_ages)
    pred_ages = np.asarray(pred_ages)

    # determine plotting range
    lo = min(true_ages.min(), pred_ages.min())
    hi = max(true_ages.max(), pred_ages.max())

    # fit regression
    slope, intercept = np.polyfit(true_ages, pred_ages, 1)
    fit_x = np.array([lo, hi])
    fit_y = slope * fit_x + intercept

    # create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(true_ages, pred_ages, alpha=0.6, edgecolors='k')
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='Ideal (y=x)')
    ax.plot(fit_x, fit_y,    'b-', linewidth=2,
            label=f'Fit: y = {slope:.2f} x + {intercept:.2f}')

    ax.set_xlabel('True Age (years)')
    ax.set_ylabel('Predicted Age (years)')
    ax.set_title('True vs. Predicted Age (Regression Fit)')
    ax.axis('square')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



