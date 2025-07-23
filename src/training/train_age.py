from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import wandb
from utils.plotting import plot_true_vs_pred_regression

############################################################
# Train Loop for Age Prediction                            #
############################################################

def train_loop(epoch, model, optimizer, device, criterion, data_loader, log_interval=200):
    # Set model to training mode
    model.train()
    total_train_loss = 0
    total_preds = []
    total_targets = []

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device).unsqueeze(1).float()

        # Reset the gradient buffers to zero
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        # Updates the gradients buffer on each parameter
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_train_loss += loss.item()
      
        total_preds.append(output.detach().cpu())
        total_targets.append(target.cpu())

    preds = torch.cat(total_preds).numpy()
    targets = torch.cat(total_targets).numpy()
    
    # Calculate metrics
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)

    # Divided by len(data_loader) because it is the sum across all batches, therefore it's divided by the number of batches.
    total_train_loss = total_train_loss / len(data_loader)

    print('loss: {:.4f}, r2: {:.4f}, mae: {:.4f}\n'.format(total_train_loss, r2, mae))
    
    return {
        "loss": total_train_loss,
        "r2": r2,
        "mae": mae,
    }

############################################################
# Validation Loop for Age Prediction                       #
############################################################

@torch.inference_mode()
def validate_loop(model, device, data_loader, criterion):
    # Put the model in eval mode, which disables
    # training specific behaviour, such as Dropout.
    model.eval()
    
    val_loss = 0
    total_preds = []
    total_targets = []
    
    for data, target in tqdm(data_loader, desc="Validation"):
        data = data.to(device)
        target = target.to(device).unsqueeze(1).float()
        output = model(data)

        val_loss += criterion(output, target).item()
        total_preds.append(output.detach().cpu())
        total_targets.append(target.detach().cpu())

    val_loss /= len(data_loader)
    preds = torch.cat(total_preds).numpy()
    targets = torch.cat(total_targets).numpy()

    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)

    
    print('\nValidation set: Average loss: {:.4f}, r2: {:.4f}, mae: {:.4f}\n'.format(val_loss, r2, mae))
    
    return {
        "loss": val_loss,
        "r2": r2,
        "mae": mae,
        "predictions": preds,
        "targets": targets,
    }

############################################################
# Function handling the full Training                      #
############################################################

def train_age(model, lr, weight_decay, epochs, train_loader, test_loader, save_path_model=None, save_path_plots=None, aug_config=None, use_wandb=False, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = model.to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    config = {
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "optiomizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "model": model.__class__.__name__,
        "augmentation": aug_config,
    }

    if use_wandb and run_name is not None:
        wandb.init(project="VT1_Age_Prediction",
                   name=run_name,
                   config=config)
        
    train_losses = []
    train_r2 = []
    train_mae = []
    val_losses = []
    val_r2 = []
    val_mae = []

    for epoch in range(1, epochs + 1):
        train_result = train_loop(epoch, cnn_model, optimizer, device, criterion, train_loader)
        train_losses.append(train_result["loss"])
        train_r2.append(train_result["r2"])
        train_mae.append(train_result["mae"])
        
        val_result = validate_loop(cnn_model, device, test_loader, criterion)
        val_losses.append(val_result["loss"])
        val_r2.append(val_result["r2"])
        val_mae.append(val_result["mae"])

        if use_wandb and run_name is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_r2": train_result["r2"],
                "train_mae": train_result["mae"],
                "val_loss": val_result["loss"],
                "val_r2": val_result["r2"],
                "val_mae": val_result["mae"],
            })

    if save_path_model is not None:
        # Save the model
        torch.save(cnn_model.state_dict(), save_path_model)
        print(f"Model saved to {save_path_model}")

    if use_wandb and run_name is not None:
        wandb.finish()

    # create plots
    result = validate_loop(cnn_model, device, test_loader, criterion)
    targets = np.array(result["targets"]).flatten()
    preds = np.array(result["predictions"]).flatten()
    plot_true_vs_pred_regression(targets, preds, save_path=save_path_plots)
