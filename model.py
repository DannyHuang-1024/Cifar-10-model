import torch.nn as nn
import torch
import torch.nn.functional as F 

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        # super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        self.conv_dropout = nn.Dropout2d(p=0.4)
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
        self.resblocks = nn.Sequential(
            *( n_blocks*[ResBlock(n_chans)] )
            )
        self.fc1 = nn.Linear(8*8*n_chans, 512)
        self.fc_dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512, 128)
        self.fc_dropout = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(128, 10)

    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)),2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8*8*self.n_chans)
        out = torch.relu(self.fc1(out))
        out = self.fc_dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out