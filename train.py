import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import datetime
import time
from sklearn.metrics import f1_score
import numpy as np

from cfDatasets import CfDatasets
from model import finalNet

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}")

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler):
    best_acc = 0.0
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_train = 0.0
        
        start_time = time.time()
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()

            loss_train += loss.item()

        avg_loss = loss_train / len(train_loader)
        
        # Validate logic
        if epoch % 2 == 0 or epoch > n_epochs - 10:
            # --- GET ACC and F1 ---
            val_acc, val_f1 = validate(model, val_loader)
            
            # --- Update Print ---
            log_str = '{} Epoch {:03d}, Loss: {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f} (Time: {:.1f}s)'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), 
                epoch, avg_loss, val_acc, val_f1, time.time() - start_time
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_cifar_model.pth")
                log_str += " [BEST]"
            
            print(log_str)
            
        else:
             print('{} Epoch {:03d}, Train Loss: {:.4f}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), epoch, avg_loss))


def validate(model, loader):
    model.eval()
    
    # Lists to store all predictions and true labels
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            # labels are usually needed on GPU for loss, 
            # but for metric calculation we will move them back to CPU later
            labels = labels.to(device) 

            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            
            # Move to CPU and convert to numpy, then extend lists
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate Accuracy manually using numpy (easier since we have the lists)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean()
    
    # Calculate Macro F1-Score
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return acc, f1

if __name__ == "__main__":
    # CONFIG
    BATCH_SIZE = 128
    N_EPOCHS = 100
    MAX_LR = 0.01

    dataset = CfDatasets()
    cifar, cifar_val = dataset.get_datasets()

    train_loader = data.DataLoader(
        cifar, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = data.DataLoader(
        cifar_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = finalNet().to(device)
    
    # Increased Weight Decay to 1e-2 to fight overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=N_EPOCHS
    )    
    
    # Label Smoothing helps generalization
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    training_loop(
        n_epochs=N_EPOCHS,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler
    )