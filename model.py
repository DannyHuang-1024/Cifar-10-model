import torch
import torch.nn as nn
import torch.nn.functional as F 

class ResBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class finalNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use 64 filters standard for ResNet18-CIFAR
        self.in_chans = 64
        
        # Initial Conv (3x3 for CIFAR, not 7x7)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout helps prevent the final layer from overfitting
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, out_chans, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_chans, out_chans, stride))
            self.in_chans = out_chans
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        out = self.dropout(out) # Applied Dropout here
        out = self.fc(out)
        return out