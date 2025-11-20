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
from model import finalNet

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}")

    
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, scheduler):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            
            # Input to GPU0
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)   # Forward
            loss = loss_fn(outputs, labels) # Cal loss

            optimizer.zero_grad() # Clear Gradient
            loss.backward() # backward calculate grad
            optimizer.step() # Update parameters base on gradient descent

            if scheduler:
                scheduler.step()

            loss_train += loss.item() 
            
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))

def validate(model:nn.Module, train_loader, val_loader):
    model.eval()
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
    # Hyperparameters
    BATCH_SIZE = 128 # Increased from 64 for better BN stats
    N_EPOCHS = 100   # 100 is usually enough for >90% with this setup
    LR = 0.01        # Peak LR for OneCycle
    

    dataset = CfDatasets()
    cifar, cifar_val = dataset.get_datasets()


    train_loader = data.DataLoader(
        cifar, 
        batch_size=BATCH_SIZE, 
        num_workers=2,
        pin_memory=True, 
        shuffle=True
    )

    model = finalNet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9,weight_decay=1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=N_EPOCHS)    
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        n_epochs=N_EPOCHS,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler
    )

    #---------------------------------VALIDATION------------------------------------------

    val_loader = data.DataLoader(
        cifar_val, 
        batch_size=BATCH_SIZE, 
        num_workers=2,
        pin_memory=True, 
        shuffle=True
    )

    train_loader = data.DataLoader(
        cifar, 
        batch_size=BATCH_SIZE, 
        num_workers=2,
        pin_memory=True, 
        shuffle=True
    )

    validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )



