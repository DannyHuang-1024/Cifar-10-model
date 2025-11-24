import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import datetime
import time

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
        
        # Validate every 2 epochs
        if epoch % 2 == 0 or epoch > n_epochs - 10:
            val_acc = validate(model, val_loader)
            
            log_str = '{} Epoch {:03d}, Train Loss: {:.4f}, Val Acc: {:.4f} (Time: {:.1f}s)'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), 
                epoch, avg_loss, val_acc, time.time() - start_time
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save best model
                torch.save(model.state_dict(), "best_cifar_model.pth")
                log_str += " [BEST]"
            
            print(log_str)
            
        else:
             print('{} Epoch {:03d}, Train Loss: {:.4f}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), epoch, avg_loss))

    print(f"Training Finished. Best Validation Accuracy: {best_acc:.4f}")

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)                
            
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
    
    return correct / total

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