import torch
import torch.nn as nn
    
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

class Age3DCNN(nn.Module):
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
        self.fc = nn.Linear(128, 1)

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



    