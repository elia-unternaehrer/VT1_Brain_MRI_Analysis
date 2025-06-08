#%%
from tqdm import tqdm
import torch
from sklearn.metrics import r2_score

#%%
def train(epoch, model, optimizer, device, criterion, data_loader, log_interval=200):
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

    r2 = r2_score(targets, preds)

    # Divided by len(data_loader) because it is the sum across all batches, therefore it's divided by the number of batches.
    total_train_loss = total_train_loss / len(data_loader)

    print('loss: {:.4f}, r2: {}\n'.format(total_train_loss, r2))
    
    return {
        "loss": total_train_loss,
        "r2": r2,
    }

#%%

@torch.inference_mode()
def validate(model, device, data_loader, criterion):
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
        total_targets.append(target.cpu())

    val_loss /= len(data_loader)
    preds = torch.cat(total_preds).numpy()
    targets = torch.cat(total_targets).numpy()

    r2 = r2_score(targets, preds)

    
    print('\nValidation set: Average loss: {:.4f}, r2: {}\n'.format(val_loss, r2))
    
    return {
        "loss": val_loss,
        "r2": r2,
        "predictions": torch.cat(total_preds),
        "targets": torch.cat(total_targets),
    }