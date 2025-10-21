import torch.nn as nn
import torch
import torch.nn.functional as F 

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        # super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        self.conv_dropout = nn.Dropout2d(p=0.3)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity = 'relu')
        nn.init.constant_(self.batch_norm.weight, 0.5)
        nn.init.zeros_(self.batch_norm.bias)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.conv_dropout(out)
        return out+x 

class finalNet(nn.Module):
    def __init__(self, n_chans=32, n_blocks=10):
        super().__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.stage1 = nn.Sequential(ResBlock(64), ResBlock(64))

        self.fmp1 = nn.FractionalMaxPool2d(kernel_size = 2, output_ratio=0.7)

        self.stage2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.transition1 = nn.Conv2d(64, 128, kernel_size=1)

        self.fmp2 = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=0.7)

        self.stage3 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.transition2 = nn.Conv2d(128, 256, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10) # Final classification layer

    
    def forward(self, out):
        # Initial Layer
        out = torch.relu(self.bn1(self.conv1(out)))

        # Stage1
        out = self.stage1(out)
        out = self.fmp1(out)
        
        # Stage2
        out = self.transition1(out)
        out = self.stage2(out)
        out = self.fmp2(out)

        # Stage3
        out = self.transition2(out)
        out = self.stage3(out)

        #Classifer
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out