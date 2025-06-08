import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
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