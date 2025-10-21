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


            loss_train += loss.item() 
            if scheduler:
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

    model = finalNet(n_chans=64).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9,weight_decay=1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    n_epochs = 200
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=n_epochs)    
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        n_epochs=n_epochs,
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



