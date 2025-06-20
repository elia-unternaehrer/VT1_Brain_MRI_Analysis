import torch
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.metrics import HausdorffDistanceMetric
import torch.nn.functional as F
import numpy as np
import warnings
import wandb
import pandas as pd
import gc

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



def train_loop(epoch, model, optimizer, criterion, data_loader, device, dice_metric, use_wandb=False):
    # set model to training mode
    model.train()
    total_loss = 0.0

    dice_metric.reset()  # Reset the Dice metric for this epoch

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
        dice_metric(pred_classes, label_cropped)

        if use_wandb and batch_idx == 0:
            # Extract one sample from the batch
            img = data[0].detach().cpu().numpy()               # (C, D, H, W)
            label = target[0].detach().cpu().numpy()            # (D, H, W)

            img = np.squeeze(img)  # (D, H, W)
            
            D, H, W = img.shape

            # undo normalization
            img = img * stds[0].item() + means[0].item()

            # scale image to [0, 255] for visualization
            img_disp = (img - img.min()) / (img.ptp() + 1e-8)
            img_disp = (img_disp * 255).astype(np.uint8)

            slices = {
                "axial":   (img_disp[D // 2, :, :], label[D // 2, :, :]),
                "coronal": (img_disp[:, H // 2, :], label[:, H // 2, :]),
                "sagittal":(img_disp[:, :, W // 2], label[:, :, W // 2]),
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
    avg_dices = dice_metric.compute().tolist()
    mean_dice_all = sum(avg_dices) / len(avg_dices)

    result = f"Loss: {total_loss:.4f}"
    for i, dice in enumerate(avg_dices):
        result += f" | Class {i} Dice: {dice:.4f}"
    result += f" | Mean Dice across all classes: {mean_dice_all:.4f}\n"
    print(result)

    return total_loss, avg_dices, mean_dice_all

def validate_loop(epoch, model, criterion, data_loader, device, dice_metric, hausdorff_metric=None):

    model.eval()
    total_loss = 0.0

    dice_metric.reset()  # Reset the Dice metric for this epoch

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
            dice_metric(pred_classes, label_cropped)

            if hausdorff_metric is not None:
                # Calculate Hausdorff distance
                pred_onehot = F.one_hot(pred_classes, num_classes=3).permute(0, 4, 1, 2, 3).float()  # [B, C, D', H', W']
                target_onehot = F.one_hot(label_cropped, num_classes=3).permute(0, 4, 1, 2, 3).float()  # [B, C, D', H', W']
                
                hausdorff_metric(pred_onehot, target_onehot)

                del pred_onehot, target_onehot

            gc.collect()  # Clear memory
            torch.cuda.empty_cache()  # Clear CUDA memory
            
    # Calculate average loss
    total_loss /= len(data_loader)

    # Calculate average Dice score
    avg_dices = dice_metric.compute().tolist()
    mean_dice_all = sum(avg_dices) / len(avg_dices)

    mean_hd95 = 0

    if hausdorff_metric is not None:
        # Calculate Hausdorff distance
        hd_result = hausdorff_metric.aggregate().cpu().numpy()
        mean_hd95 = np.nanmean(hd_result)
        hausdorff_metric.reset()

    result = f"Loss: {total_loss:.4f}"
    for i, dice in enumerate(avg_dices):
        result += f" | Class {i} Dice: {dice:.4f}"
    result += f" | Mean Dice across all classes: {mean_dice_all:.4f}\n"
    if hausdorff_metric is not None:
        result += f" | Mean HD95: {mean_hd95:.4f}\n"
    print(result)

    return total_loss, avg_dices, mean_dice_all, mean_hd95

def train_segmentation(model, lr, weight_decay, epochs, train_loader, test_loader, weights=None, use_hausdorff=False, save_path=None, use_wandb=False, run_name=None, aug_config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = model.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize Dice metrics
    dice_metric = DiceScore(num_classes=3, include_background=True, average="none", input_format='index', zero_division=1.0).to(device)

    if use_hausdorff:
        # Initialize Hausdorff distance metric
        hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction='mean')

    config={
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "scheduler": "ReduceLROnPlateau",
            "augmentation": aug_config
        }

    if use_wandb and run_name is not None:
        wandb.init(project="VT1_Hippocampus_Segmentation", name=run_name,
        config=config)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    # If weights are provided, convert them to a tensor and move to device
    if weights is not None:
        # Convert weights to tensor and move to device
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, epochs + 1):
        
        calc_hausdorff = (epoch % 5 == 0 or epoch == 1 or epoch == epochs) and use_hausdorff

        # Training
        train_loss, train_avg_dices, train_mean_dice_all = train_loop(epoch, unet, optimizer, criterion, train_loader, device, dice_metric, use_wandb=(use_wandb and run_name is not None))
        
        torch.cuda.empty_cache()

        
        # Validation
        val_loss, val_avg_dices, val_mean_dice_all, val_mean_hd95 = validate_loop(epoch, unet, criterion, test_loader, device, dice_metric, hausdorff_metric=(hausdorff_metric if calc_hausdorff else None))
        

        torch.cuda.empty_cache()
    
        # Update the learning rate
        scheduler.step(val_loss)

        # Early stopping logic
        if  val_loss < best_val_loss + 1e-5:
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model checkpoint
            if save_path is not None:
                torch.save(unet.state_dict(), save_path)
                print(f"New best model saved with val loss {val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience and False:
            print("Early stopping triggered.")
            break
    
        # If using wandb, log the final results
        if use_wandb and run_name is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice_mean": train_mean_dice_all,
                "val_loss": val_loss,
                "val_dice_mean": val_mean_dice_all,
                "val_hd95": val_mean_hd95 if calc_hausdorff else None,
                **{f"train_dice_class_{i}": v for i, v in enumerate(train_avg_dices)},
                **{f"val_dice_class_{i}": v for i, v in enumerate(val_avg_dices)}
            })

    
    if save_path is not None:
        torch.save(unet.state_dict(), save_path)
        
    # If using wandb, finish the run
    if use_wandb and run_name is not None:
        wandb.finish()

