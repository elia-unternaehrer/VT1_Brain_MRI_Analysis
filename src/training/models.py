import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
# Sex Classification Model                                 #
############################################################
    
class Sex3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.instancenorm1 = nn.InstanceNorm3d(16, affine=True)
        self.pool1 = nn.MaxPool3d(2)  # 256 => 128

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.instancenorm2 = nn.InstanceNorm3d(32, affine=True)
        self.pool2 = nn.MaxPool3d(2)  # 128 => 64

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.instancenorm3 = nn.InstanceNorm3d(64, affine=True)
        self.pool3 = nn.MaxPool3d(2)  # 64 => 32

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.instancenorm4 = nn.InstanceNorm3d(128, affine=True)
        self.pool4 = nn.MaxPool3d(2)   # 32 => 16

        # Adaptive pooling to reduce the output to a fixed size
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.instancenorm1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.instancenorm2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.instancenorm3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.instancenorm4(self.conv4(x)))
        x = self.pool4(x)

        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
    
############################################################
# Age Prediction Model Variants                            #
############################################################

class Age3DCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.instancenorm1 = nn.InstanceNorm3d(16, affine=True)
        self.pool1 = nn.MaxPool3d(2)  # 256 => 128

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.instancenorm2 = nn.InstanceNorm3d(32, affine=True)
        self.pool2 = nn.MaxPool3d(2)  # 128 => 64

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.instancenorm3 = nn.InstanceNorm3d(64, affine=True)
        self.pool3 = nn.MaxPool3d(2)  # 64 => 32

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.instancenorm4 = nn.InstanceNorm3d(128, affine=True)
        self.pool4 = nn.MaxPool3d(2)   # 32 => 16

        # Adaptive pooling to reduce the output to a fixed size
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 1)  # Output layer for age

    def forward(self, x):
        x = torch.relu(self.instancenorm1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.instancenorm2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.instancenorm3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.instancenorm4(self.conv4(x)))
        x = self.pool4(x)

        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc1(x)

class Age3DCNNExtended(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.instancenorm1 = nn.InstanceNorm3d(16, affine=True)
        self.pool1 = nn.MaxPool3d(2)  # 256 => 128

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.instancenorm2 = nn.InstanceNorm3d(32, affine=True)
        self.pool2 = nn.MaxPool3d(2)  # 128 => 64

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.instancenorm3 = nn.InstanceNorm3d(64, affine=True)
        self.pool3 = nn.MaxPool3d(2)  # 64 => 32

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.instancenorm4 = nn.InstanceNorm3d(128, affine=True)
        self.pool4 = nn.MaxPool3d(2)   # 32 => 16

        # Adaptive pooling to reduce the output to a fixed size
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.instancenorm1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.instancenorm2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.instancenorm3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.instancenorm4(self.conv4(x)))
        x = self.pool4(x)

        x = self.adapt_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)
    
############################################################
# Hippocampus Segmentation Model Variants                  #
############################################################

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
    
class ConvBlockBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockBatchNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvBlockInstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockInstanceNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3),
            nn.InstanceNorm3d(out_channels),
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

class UNet3DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super().__init__()

        # Encoder 1
        self.enc1 = ConvBlockBatchNorm(in_channels, base_filters)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 2
        self.enc2 = ConvBlockBatchNorm(base_filters, base_filters * 2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 3
        self.enc3 = ConvBlockBatchNorm(base_filters * 2, base_filters * 4)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlockBatchNorm(base_filters * 4, base_filters * 8)

        # Decoder 1
        self.upconv1 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlockBatchNorm(base_filters * 8, base_filters * 4)

        # Decoder 2
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlockBatchNorm(base_filters * 4, base_filters * 2)

        # Decoder 3
        self.upconv3 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec3 = ConvBlockBatchNorm(base_filters * 2, base_filters)

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
    
class UNet3DInstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super().__init__()

        # Encoder 1
        self.enc1 = ConvBlockInstanceNorm(in_channels, base_filters)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 2
        self.enc2 = ConvBlockInstanceNorm(base_filters, base_filters * 2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder 3
        self.enc3 = ConvBlockInstanceNorm(base_filters * 2, base_filters * 4)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlockInstanceNorm(base_filters * 4, base_filters * 8)

        # Decoder 1
        self.upconv1 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlockInstanceNorm(base_filters * 8, base_filters * 4)

        # Decoder 2
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlockInstanceNorm(base_filters * 4, base_filters * 2)

        # Decoder 3
        self.upconv3 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec3 = ConvBlockInstanceNorm(base_filters * 2, base_filters)

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
    
############################################################
# Dice Loss and Combined Loss                              #
############################################################

class MultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index # ingnore background class
        self.smooth = smooth # prevent division by zero

    def forward(self, inputs, targets):
        """
        inputs: raw logits from model, shape [B, C, D, H, W]
        targets: class labels, shape [B, D, H, W]
        """
        num_classes = inputs.shape[1] # get numb of classes
        inputs = F.softmax(inputs, dim=1) # convert output into probabilities

        # One-hot encode targets to shape [B, C, D, H, W]
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # [B, D, H, W, C]
        targets_onehot = targets_onehot.permute(0, -1, *range(1, targets.ndim)).float() # [B, C, D, H, W]

        dice = 0.0
        count = 0

        # compute dice per classe
        for c in range(num_classes):
            if c == self.ignore_index:
                continue # ignore bg class
            input_c = inputs[:, c] # props for class c
            target_c = targets_onehot[:, c] # label for class c

            # calculate dice components
            intersection = (input_c * target_c).sum()
            union = input_c.sum() + target_c.sum()

            # calculate dice for class and add
            dice += (2. * intersection + self.smooth) / (union + self.smooth)
            count += 1

        # free memory
        del targets_onehot

        # return avg dice loss
        return 1.0 - dice / count

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=0, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = MultiClassDiceLoss(ignore_index=ignore_index)
        self.alpha = alpha

    def forward(self, inputs, targets):
        # compute both loss
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        # return weighted combination
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss