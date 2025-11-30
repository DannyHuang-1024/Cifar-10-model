import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Import project modules
from model import finalNet
from cfDatasets import CfDatasets

# --- Configuration ---
MODEL_PATH = './best_cifar_model.pth'
OUTPUT_DIR = './plots'  # Save to 'plots' to match README structure
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 Classes
CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def generate_matrix():
    # 1. Prepare Data
    print("Loading validation dataset...")
    dataset_handler = CfDatasets()
    _, val_dataset = dataset_handler.get_datasets()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Load Model
    print(f"Loading model from: {MODEL_PATH} ...")
    model = finalNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Error: Model weights not found! Please run train.py first.")
        return
    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []

    print("Starting inference on validation set...")
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # 5. Plotting
    # Increase figure size for clarity
    plt.figure(figsize=(10, 8)) 
    
    # Draw Heatmap
    # cmap='Blues' is standard for academic papers
    # annot_kws={"size": 12} sets the font size of the numbers inside the boxes
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=CLASSES, yticklabels=CLASSES,
                     annot_kws={"size": 12})
    
    # Labels and Title
    plt.title('CIFAR-10 Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Adjust tick label size
    plt.xticks(fontsize=11, rotation=45) 
    plt.yticks(fontsize=11, rotation=0)

    # 6. Save Files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Path definitions
    pdf_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')

    # Save as PDF (Vector graphic for LaTeX Report)
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved PDF matrix to: {pdf_path}")

    # Save as PNG (Bitmap for README/Web)
    plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
    print(f"Saved PNG matrix to: {png_path}")
    
    # plt.show() # Uncomment if you want to see the plot immediately

if __name__ == '__main__':
    generate_matrix()