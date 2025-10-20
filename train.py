# Thrid Party Library
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as data


import datetime

# Self Definition Module
from cfDatasets import CfDatasets

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}")

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        # super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        self.conv_dropout = nn.Dropout2d(p=0.2)
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
            *( n_blocks*[ResBlock(n_chans=32)] )
            )
        self.fc1 = nn.Linear(8*8*n_chans, 64)
        # self.fc_dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)),2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8*8*self.n_chans)
        out = torch.relu(self.fc1(out))
        # out = self.fc_dropout(out)
        out = self.fc2(out)
        return out
    
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, scheduler):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            
            # Input to GPU
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)   # Forward
            loss = loss_fn(outputs, labels) # Cal loss

            optimizer.zero_grad() # Clear Gradient
            loss.backward() # backward calculate grad
            optimizer.step() # Update parameters base on gradient descent


            loss_train += loss.item() 
        scheduler.step()

        
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))

def validate(model:nn.Module, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:

                # Input to GPU
                imgs = imgs.to(device)
                labels = labels.to(device)                
                
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {} : {:.2f}".format(
            name, correct / total
        ))


if __name__ == "__main__":
    

    dataset = CfDatasets()
    cifar, cifar_val = dataset.get_datasets()


    train_loader = data.DataLoader(
        cifar, batch_size=64, shuffle=True
    )

    model = finalNet(n_blocks=20).to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-3, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) 
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        n_epochs=200,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler
    )

    #---------------------------------VALIDATION------------------------------------------

    val_loader = data.DataLoader(
        cifar_val, batch_size=64, shuffle=False
    )

    train_loader = data.DataLoader(
        cifar, batch_size=64, shuffle=False
    )

    validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )



