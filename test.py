import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 引入你的模型定义
from model import finalNet 

# --- 配置参数 ---
MODEL_PATH = './best_cifar_model.pth'  # 模型权重路径
DATA_DIR = '/home/danny/data/test'        # 测试图片根目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 类别顺序 (必须与训练时一致)
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']
# 建立 类别名 -> 索引 的映射，例如 {'plane': 0, 'car': 1 ...}
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

def load_model():
    model = finalNet()
    # map_location='cpu' 保证在无显卡的机器上也能加载 GPU 训练的模型
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # 开启评估模式 (关闭 Dropout)
    return model

def get_image_paths(root_dir):
    """
    遍历文件夹，返回 [(图片路径, 真实标签名), ...]
    """
    image_list = []
    # 遍历 root_dir 下的所有文件夹
    for label_name in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label_name)
        # 确保是文件夹且是我们的有效类别
        if os.path.isdir(label_dir) and label_name in CLASS_TO_IDX:
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(label_dir, fname)
                    image_list.append((file_path, label_name))
    return image_list

def predict_and_plot(model, image_list):
    """
    预测并绘图的核心函数
    """
    # 1. 定义预处理 (必须与训练时一致)
    # 输入模型的 transform
    infer_transform = transforms.Compose([
        transforms.Resize((32, 32)), # 模型需要 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # 2. 设置绘图网格
    num_imgs = len(image_list)
    if num_imgs == 0:
        print("未找到图片，请检查路径。")
        return

    # 计算行数和列数 (尽量接近正方形)
    cols = int(math.ceil(math.sqrt(num_imgs)))
    rows = int(math.ceil(num_imgs / cols))
    
    # 创建大图，figsize 可以根据图片数量调整
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten() if num_imgs > 1 else [axes]

    print(f"开始预测 {num_imgs} 张图片...")

    with torch.no_grad():
        for i, (img_path, true_label_name) in enumerate(image_list):
            ax = axes[i]
            
            # --- A. 读取与预处理 ---
            original_img = Image.open(img_path).convert('RGB')
            
            # 准备输入 Tensor
            img_tensor = infer_transform(original_img)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE) # 加 Batch 维 [1, 3, 32, 32]
            
            # --- B. 模型预测 ---
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1) # 获取概率
            pred_score, pred_idx = torch.max(probs, 1)
            
            pred_idx = pred_idx.item()
            pred_label_name = CLASSES[pred_idx]
            confidence = pred_score.item()
            
            # --- C. 判断正误 ---
            is_correct = (pred_label_name == true_label_name)
            
            # --- D. 绘图与染色 ---
            # 显示原图 (resize 一下方便显示，不然太小)
            display_img = original_img.resize((128, 128)) 
            ax.imshow(display_img)
            
            # 关键：根据正误添加颜色遮罩
            if is_correct:
                # 绿色遮罩 (Green), alpha=0.3 (半透明)
                rect = patches.Rectangle((0,0), 128, 128, linewidth=0, 
                                         edgecolor='none', facecolor='green', alpha=0.3)
                color_code = 'green'
                status_text = "Correct"
            else:
                # 红色遮罩 (Red)
                rect = patches.Rectangle((0,0), 128, 128, linewidth=0, 
                                         edgecolor='none', facecolor='red', alpha=0.3)
                color_code = 'red'
                status_text = "Wrong"
            
            ax.add_patch(rect) # 添加遮罩层

            # 设置标题 (预测: 概率 / 真实)
            title_text = f"Pred: {pred_label_name} ({confidence:.2f})\nTrue: {true_label_name}"
            ax.set_title(title_text, color=color_code, fontsize=10, fontweight='bold')
            ax.axis('off') # 隐藏坐标轴

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('result_visualization.png') # 保存结果图
    print("预测完成！结果已保存为 result_visualization.png")
    plt.show()

if __name__ == '__main__':
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
    else:
        net = load_model()
        
        # 2. 获取图片列表
        imgs = get_image_paths(DATA_DIR)
        
        # 3. 预测并画图
        predict_and_plot(net, imgs)


#/home/danny/data/test