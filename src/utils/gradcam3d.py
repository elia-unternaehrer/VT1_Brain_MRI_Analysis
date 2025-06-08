import torch
import torch.nn.functional as F
import utils.plotting as plotting
import torch.nn as nn
import numpy as np

class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)  # Forward pass
        class_score = output[:, target_class]
        class_score.backward()

        # Get mean gradient across D, H, W
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)  # shape [B, C, 1, 1, 1]

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # shape [B, 1, D, H, W]
        cam = F.relu(cam)

        # Normalize
        cam = cam - np.percentile(cam.cpu().numpy(), 95)
        cam -= cam.min()
        #cam /= cam.max() + 1e-8

        return cam  # shape [1, 1, D, H, W]
    
def gradcam_visualization(model, test_loader, mode="nearest", layer=None, num_samples_per_class=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = model.to(device)

    # Automatically find all conv layers (assumes names like 'conv1', 'conv2', ...)
    conv_layers = [(name, layer) for name, layer in trained_model.named_modules() if isinstance(layer, nn.Conv3d)]
    conv_layers = sorted(conv_layers, key=lambda x: x[0])  # Sort by name (conv1 < conv2 < ...)

    if isinstance(layer, int):
        if layer <= 0 or layer > len(conv_layers):
            raise ValueError(f"Invalid conv layer index {layer}. Model has {len(conv_layers)} conv layers.")
        target_layer = conv_layers[layer - 1][1]
    elif isinstance(layer, nn.Module):
        target_layer = layer
    else:
        # Default: last conv layer
        target_layer = conv_layers[-1][1]
        print(f"Using last conv layer: {target_layer}")

    # Wrap it with GradCAM
    gradcam = GradCAM3D(trained_model, target_layer)

    # Load a sample MRI image
    data_iter = iter(test_loader)
    male = 0
    female = 0

    while male < num_samples_per_class or female < num_samples_per_class:
        input_tensor, label = next(data_iter)

        if label.item() == 0 and male < num_samples_per_class:
            male += 1
        elif label.item() == 1 and female < num_samples_per_class:
            female += 1
        else:
            continue 


        # Input MRI tensor: [1, 1, 128, 128, 128]
        input_tensor = input_tensor.to(device)
        target_class = torch.argmax(trained_model(input_tensor), dim=1).item()

        # Generate CAM
        cam = gradcam.generate(input_tensor, target_class)  # shape: [1, 1, 16, 16, 16]

        # Upsample to 128³ for visualization
        if mode == "nearest":
            cam_upsampled = F.interpolate(cam, size=(128, 128, 128), mode='nearest')
        elif mode == "trilinear":
            cam_upsampled = F.interpolate(cam, size=(128, 128, 128), mode='trilinear', align_corners=False)

        plotting.plot_mri_with_cam(input_tensor, cam_upsampled, label.item(), target_class)
