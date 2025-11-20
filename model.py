import torch.nn as nn
import torch
import torch.nn.functional as F 

class ResBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        
        # Shortcut handling: If input shape != output shape, we need to project x
        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Skip Connection
        out = F.relu(out)
        return out 

class finalNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial Stage
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers
        # Layer 1: 64 channels, same size
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        
        # Layer 2: 128 channels, downsample (stride=2)
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1)
        )
        
        # Layer 3: 256 channels, downsample (stride=2)
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1)
        )
        
        # Layer 4: 512 channels, downsample (stride=2)
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out        