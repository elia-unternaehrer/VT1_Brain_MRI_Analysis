import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
def crop_tensor(enc_feat, target):
        """
        Crop encoder feature map to match the spatial size of target feature map.
        Assumes input shapes are (B, C, D, H, W).
        """
        # get the difference in dimensions between encoder feature map and decoder feature map
        diff_depth = enc_feat.size(2) - target.size(2)
        diff_height = enc_feat.size(3) - target.size(3)
        diff_width = enc_feat.size(4) - target.size(4)

        # crop the encoder feature map to match the spatial size of the target feature map
        return enc_feat[:, :,
            diff_depth // 2 : enc_feat.size(2) - (diff_depth - diff_depth // 2),
            diff_height // 2 : enc_feat.size(3) - (diff_height - diff_height // 2),
            diff_width // 2 : enc_feat.size(4) - (diff_width - diff_width // 2)
        ]
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super().__init__()

        # Encoder 1
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 2
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 3
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

        # Decoder 1
        self.upconv1 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 8, base_filters * 4)

        # Decoder 2
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        # Decoder 3
        self.upconv3 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 2, base_filters)

        # Output layer
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder 1
        enc1 = self.enc1(x)
        enc1_pool = self.maxpool1(enc1)

        # Encoder 2
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.maxpool2(enc2)

        # Encoder 3
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.maxpool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)

        # Decoder 1
        dec1_up = self.upconv1(bottleneck)
        cropped_enc3 = crop_tensor(enc3, dec1_up)
        assert dec1_up.shape[2:] == cropped_enc3.shape[2:], "Cropping failed (enc2 -> dec1)"
        dec1 = torch.cat((dec1_up, cropped_enc3), dim=1)        
        dec1_out = self.dec1(dec1)

        # Decoder 2
        dec2_up = self.upconv2(dec1_out)
        cropped_enc2 = crop_tensor(enc2, dec2_up)
        assert dec2_up.shape[2:] == cropped_enc2.shape[2:], "Cropping failed (enc1 -> dec2)"
        dec2 = torch.cat((dec2_up, cropped_enc2), dim=1)
        dec2_out = self.dec2(dec2)

        # Decoder 3
        dec3_up = self.upconv3(dec2_out)
        cropped_enc1 = crop_tensor(enc1, dec3_up)
        assert dec3_up.shape[2:] == cropped_enc1.shape[2:], "Cropping failed (input -> dec3)"
        dec3 = torch.cat((dec3_up, cropped_enc1), dim=1)
        dec3_out = self.dec3(dec3)

        # Output layer
        out = self.out_conv(dec3_out)

        return out
    

class MultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: raw logits from model, shape [B, C, D, H, W]
        targets: class labels, shape [B, D, H, W]
        """
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets to shape [B, C, ...]
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # [B, ..., C]
        targets_onehot = targets_onehot.permute(0, -1, *range(1, targets.ndim)).float()

        dice = 0.0
        count = 0

        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            input_c = inputs[:, c]
            target_c = targets_onehot[:, c]

            intersection = (input_c * target_c).sum()
            union = input_c.sum() + target_c.sum()
            dice += (2. * intersection + self.smooth) / (union + self.smooth)
            count += 1

        del targets_onehot
        
        return 1.0 - dice / count

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=0, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = MultiClassDiceLoss(ignore_index=ignore_index)
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
