import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import your model definition
from model import finalNet 

# --- Configuration parameters ---
MODEL_PATH = './best_cifar_model.pth'  # Path to the model weights
DATA_DIR = '/home/danny/data/test'     # Root directory of test images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class order (must match the order used during training)
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Build mapping: class_name -> index, e.g., {'plane': 0, 'car': 1, ...}
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

def load_model():
    model = finalNet()
    # map_location='cpu' ensures that GPU-trained weights can be loaded on CPU-only machines
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()  # Switch to evaluation mode (disable dropout)
    return model

def get_image_paths(root_dir):
    """
    Traverse directory and return a list of:
    [(image_path, true_label_name), ...]
    """
    image_list = []
    # Traverse all subfolders under root_dir
    for label_name in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label_name)
        # Ensure it is a directory and is a valid class
        if os.path.isdir(label_dir) and label_name in CLASS_TO_IDX:
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(label_dir, fname)
                    image_list.append((file_path, label_name))
    return image_list

def predict_and_plot(model, image_list):
    """
    Core function for predicting and visualizing results.
    """
    # 1. Define preprocessing (must match training)
    infer_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Model expects 32×32 input
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # 2. Setup the plotting grid
    num_imgs = len(image_list)
    if num_imgs == 0:
        print("No images found. Please check your path.")
        return

    # Compute number of rows/columns to make grid close to square
    cols = int(math.ceil(math.sqrt(num_imgs)))
    rows = int(math.ceil(num_imgs / cols))
    
    # Create the figure; adjust figsize based on image count
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if num_imgs > 1 else [axes]

    print(f"Starting prediction on {num_imgs} images...")

    with torch.no_grad():
        for i, (img_path, true_label_name) in enumerate(image_list):
            ax = axes[i]
            
            # --- A. Read and preprocess image ---
            original_img = Image.open(img_path).convert('RGB')
            
            # Prepare input tensor
            img_tensor = infer_transform(original_img)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # Add batch dim: [1,3,32,32]
            
            # --- B. Model prediction ---
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)  # Get probabilities
            pred_score, pred_idx = torch.max(probs, 1)
            
            pred_idx = pred_idx.item()
            pred_label_name = CLASSES[pred_idx]
            confidence = pred_score.item()
            
            # --- C. Compare with ground truth ---
            is_correct = (pred_label_name == true_label_name)
            
            # --- D. Plot and color overlay ---
            # Resize image for clearer display
            display_img = original_img.resize((128, 128))
            ax.imshow(display_img)
            
            # Add semi-transparent green/red overlay
            if is_correct:
                rect = patches.Rectangle(
                    (0,0), 128, 128, linewidth=0,
                    edgecolor='none', facecolor='green', alpha=0.3
                )
                color_code = 'green'
                status_text = "Correct"
            else:
                rect = patches.Rectangle(
                    (0,0), 128, 128, linewidth=0,
                    edgecolor='none', facecolor='red', alpha=0.3
                )
                color_code = 'red'
                status_text = "Wrong"
            
            ax.add_patch(rect)

            # Title showing prediction/confidence/ground truth
            title_text = (
                f"Pred: {pred_label_name} ({confidence:.2f})\n"
                f"True: {true_label_name}"
            )
            ax.set_title(title_text, color=color_code, fontsize=10, fontweight='bold')
            ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('result_visualization.png')  # Save output
    print("Prediction completed! Saved as result_visualization.png")
    plt.show()

if __name__ == '__main__':
    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model file not found: {MODEL_PATH}")
    else:
        net = load_model()
        
        # 2. Load images
        imgs = get_image_paths(DATA_DIR)
        
        # 3. Predict and visualize
        predict_and_plot(net, imgs)
