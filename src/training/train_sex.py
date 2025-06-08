#%%
from tqdm import tqdm
import torch
import utils.plotting as plotting
from torch.utils.data import DataLoader

#%%
def train_loop(epoch, model, optimizer, device, criterion, data_loader):
    # Set model to training mode
    model.train()
    total_train_loss = 0
    total_correct = 0

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch}")):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

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
      
        _, preds = torch.max(output, dim=1)       # preds shape (B,) with values 0 or 1
        total_correct += (preds == target).sum().item()
    
    
    # Divided by len(data_loader.dataset) because it's the number of correct predictions in total.
    accuracy_train = total_correct / len(data_loader.dataset)

    # Divided by len(data_loader) because it is the sum across all batches, therefore it's divided by the number of batches.
    total_train_loss = total_train_loss / len(data_loader)

    print('loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(total_train_loss, total_correct, len(data_loader.dataset), 100 * accuracy_train))
    
    return {
        "loss": total_train_loss,
        "accuracy": accuracy_train,
    }

#%%

@torch.inference_mode()
def validate_loop(model, device, data_loader, criterion):
    # Put the model in eval mode, which disables
    # training specific behaviour, such as Dropout.
    model.eval()
    
    val_loss = 0
    total_correct = 0
    total_preds = []
    total_targets = []
    total_probs = []
    total_outputs = []
    
    for data, target in tqdm(data_loader, desc="Validation"):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).item()
        
        probs = torch.softmax(output, dim=1)
        _, preds = torch.max(output, dim=1)       # preds shape (B,) with values 0 or 1
        total_correct += (preds == target).sum().item()


        # Keep the predictions and targets to inspect later on.
        # .detach() decouples it from the computational graph
        # and .cpu() ensures it's on the CPU, since it shouldn't
        # occupy any unnecesary GPU memory.
        total_preds.append(preds.detach().cpu())
        # target isn't tracked in the computational graph
        # (it's a leaf with requires_grad=False), adding the
        # .detach() doesn't do anything, but if in doubt, just add it.
        total_targets.append(target.cpu())
        total_probs.append(probs.detach().cpu())
        total_outputs.append(output.detach().cpu())

    val_loss /= len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, total_correct, len(data_loader.dataset), 100 * accuracy))
    
    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "predictions": torch.cat(total_preds),
        "probabilities": torch.cat(total_probs),
        "targets": torch.cat(total_targets),
        "outputs": torch.cat(total_outputs),
    }

#%%

def train_sex(model, lr, weight_decay, epochs, train_loader, test_loader, save_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model = model.to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Keep track of stats to plot them
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        train_result = train_loop(epoch, cnn_model, optimizer, device, criterion, train_loader)
        train_losses.append(train_result["loss"])
        train_accuracies.append(train_result["accuracy"])
        
        val_result = validate_loop(cnn_model, device, test_loader, criterion)
        val_losses.append(val_result["loss"])
        val_accuracies.append(val_result["accuracy"])

    print(f"Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    plotting.plot_results(train_losses, train_accuracies, val_losses, val_accuracies)

    if save_path is not None:
        # Save the model
        torch.save(cnn_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
