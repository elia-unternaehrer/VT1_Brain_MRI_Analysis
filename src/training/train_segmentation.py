import torch
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
from utils.plotting import plot_segmentation_results
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.metrics import HausdorffDistanceMetric
import torch.nn.functional as F
import numpy as np
import warnings
import wandb

warnings.filterwarnings(
    "ignore",
    message=r"the (ground truth|prediction) of class \d+ is all 0, this may result in nan/inf distance\.",
    category=UserWarning
)

def center_crop_label(label, target_shape):
    """
    Center-crop a label tensor to match the target spatial shape.
    Assumes label is [B, D, H, W] and target_shape is [B, C, D', H', W']
    """
    _, D, H, W = label.shape
    _, _, D_t, H_t, W_t = target_shape

    d_start = (D - D_t) // 2
    h_start = (H - H_t) // 2
    w_start = (W - W_t) // 2

    return label[:, d_start:d_start + D_t,
                    h_start:h_start + H_t,
                    w_start:w_start + W_t]

def train_loop(epoch, model, optimizer, criterion, data_loader, device, use_wandb=False):
    # set model to training mode
    model.train()
    total_loss = 0.0

    # Initialize Dice metrics
    dice_metric = DiceScore(num_classes=3, include_background=True, average="none", input_format='index', zero_division=1.0).to(device)

    dice_sums = torch.zeros(3).to(device)

    for batch_idx, (data, target, means, stds) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Reset the gradient buffers to zero
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Crop the label to match the output shape
        label_cropped = center_crop_label(target, output.shape)

        assert output.shape[2:] == label_cropped.shape[1:], "Label cropping failed"

        # Calculate loss
        loss = criterion(output, label_cropped)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()

        # Calculate Dice score
        pred_classes = torch.argmax(output, dim=1)
        batch_dice = dice_metric(pred_classes, label_cropped)  # Tensor [3] (per class)
        dice_sums += batch_dice

        if use_wandb and batch_idx == 0:
            # Extract one sample from the batch
            img = data[0].detach().cpu().numpy()               # (C, D, H, W)
            label = target[0].detach().cpu().numpy()            # (D, H, W)

            img = np.squeeze(img)  # (D, H, W)
            
            D, H, W = img.shape

            # undo normalization
            img = img * stds[0].item() + means[0].item()  # Assuming means and stds are scalars for simplicity

            slices = {
                "axial":   (img[D // 2, :, :], label[D // 2, :, :]),
                "coronal": (img[:, H // 2, :], label[:, H // 2, :]),
                "sagittal":(img[:, :, W // 2], label[:, :, W // 2]),
            }

            for plane, (img_slice, label_slice) in slices.items():
                
                
                wandb.log({
                            f"{plane}_overlay": wandb.Image(
                                img_slice,
                                masks={"ground_truth": {"mask_data": label_slice}},
                                caption=f"{plane} Overlay"
                            )
                        }, step=epoch)

    # Calculate average loss
    total_loss /= len(data_loader)

    # Calculate average Dice score
    avg_dices = (dice_sums / len(data_loader)).tolist()
    mean_dice_all = sum(avg_dices) / len(avg_dices)

    result = f"Loss: {total_loss:.4f}"
    for i, dice in enumerate(avg_dices):
        result += f" | Class {i} Dice: {dice:.4f}"
    result += f" | Mean Dice across all classes: {mean_dice_all:.4f}\n"
    print(result)

    return total_loss, avg_dices, mean_dice_all

def validate_loop(epoch, model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0

    # Initialize Dice metrics
    dice_metric = DiceScore(num_classes=3, include_background=True, average="none", input_format='index', zero_division=1.0)
    dice_sums = torch.zeros(3).to(device)

    # Initialize Hausdorff distance metric
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction='mean')

    with torch.no_grad():  # Disable gradient computation for validation
        for data, target,_ ,_ in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
            data = data.to(device)
            target = target.to(device)

            # Pass data through the network
            output = model(data)

            # Crop the label to match the output shape
            label_cropped = center_crop_label(target, output.shape)  # [B, D', H', W']

            assert output.shape[2:] == label_cropped.shape[1:], "Label cropping failed"

            # Calculate loss
            loss = criterion(output, label_cropped)
            total_loss += loss.item()

            # Calculate Dice score
            pred_classes = torch.argmax(output, dim=1)
            batch_dice = dice_metric(pred_classes, label_cropped)
            dice_sums += batch_dice

            # Calculate Hausdorff distance
            pred_onehot = F.one_hot(pred_classes, num_classes=3).permute(0, 4, 1, 2, 3).float()  # [B, C, D', H', W']
            target_onehot = F.one_hot(label_cropped, num_classes=3).permute(0, 4, 1, 2, 3).float()  # [B, C, D', H', W']
            
            hausdorff_metric(pred_onehot, target_onehot)

    # Calculate average loss
    total_loss /= len(data_loader)

    # Calculate average Dice score
    avg_dices = (dice_sums / len(data_loader)).tolist()
    mean_dice_all = sum(avg_dices) / len(avg_dices)

    # Calculate Hausdorff distance
    hd_result = hausdorff_metric.aggregate().cpu().numpy()
    mean_hd95 = np.nanmean(hd_result)

    result = f"Loss: {total_loss:.4f}"
    for i, dice in enumerate(avg_dices):
        result += f" | Class {i} Dice: {dice:.4f}"
    result += f" | Mean Dice across all classes: {mean_dice_all:.4f}\n"
    result += f" | Mean HD95: {mean_hd95:.4f}\n"
    print(result)

    return total_loss, avg_dices, mean_dice_all, mean_hd95

def train_segmentation(model, lr, weight_decay, epochs, train_loader, test_loader, weights=None, save_path=None, use_wandb=False, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = model.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if use_wandb and run_name is not None:
        wandb.init(project="VT1_Hippocampus_Segmentation", name=run_name,
        config={
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "scheduler": "ReduceLROnPlateau"
        })

    best_val_dice = float(0)
    best_model_epoch = 0
    patience = 10
    patience_counter = 0

    # If weights are provided, convert them to a tensor and move to device
    if weights is not None:
        # Convert weights to tensor and move to device
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # Keep track of stats to plot them
    train_losses = []
    train_dices = []
    train_mean_dices = []
    val_losses = []
    val_dices = []
    val_mean_dices = []
    val_hd95s = []

    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_avg_dices, train_mean_dice_all = train_loop(epoch, unet, optimizer, criterion, train_loader, device, use_wandb=use_wandb)
        train_losses.append(train_loss)
        train_dices.append(train_avg_dices)
        train_mean_dices.append(train_mean_dice_all)
        
        # Validation
        val_loss, val_avg_dices, val_mean_dice_all, val_mean_hd95 = validate_loop(epoch, unet, criterion, test_loader, device)
        val_losses.append(val_loss)
        val_dices.append(val_avg_dices)
        val_mean_dices.append(val_mean_dice_all)
        val_hd95s.append(val_mean_hd95)

        # Update the learning rate
        scheduler.step(val_loss)

        # Early stopping logic
        if  val_mean_dice_all > best_val_dice + 1e-5:
            best_val_dice = val_mean_dice_all
            patience_counter = 0
            best_model_epoch = epoch - 1

            # Save the best model checkpoint
            if save_path is not None:
                torch.save(unet.state_dict(), save_path)
                print(f"New best model saved with val loss {val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        
        # If using wandb, log the final results
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice_mean": train_mean_dice_all,
                "val_loss": val_loss,
                "val_dice_mean": val_mean_dice_all,
                "val_hd95": val_mean_hd95,
                **{f"train_dice_class_{i}": v for i, v in enumerate(train_avg_dices)},
                **{f"val_dice_class_{i}": v for i, v in enumerate(val_avg_dices)}
            })

    # print final results
    print(f"Training:")
    print(f"    - Loss: {train_losses[best_model_epoch]:.4f}")
    for i, dice in enumerate(train_dices[best_model_epoch]):
        print(f"    - Class {i} Dice: {dice:.4f}")
    print(f"    - Mean Dice across all classes: {train_mean_dices[best_model_epoch]:.4f}\n")

    print(f"Validation")
    print(f"    - Loss: {val_losses[best_model_epoch]:.4f}")
    for i, dice in enumerate(val_dices[best_model_epoch]):
        print(f"    - Class {i} Dice: {dice:.4f}")
    print(f"    - Mean Dice across all classes: {val_mean_dices[best_model_epoch]:.4f}\n")
    print(f"    - Mean HD95: {val_hd95s[best_model_epoch]:.4f}\n")
