import torch
from torchvision import transforms
from torchvision import datasets

class CfDatasets:
    def __init__(self):
        data_path = r"./data"
        
        # Standard CIFAR-10 Mean/Std
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        # === STRONG AUGMENTATION ===
        self.train_transform = transforms.Compose([
            # 1. Padding 4 pixels helps shift the image around so the model
            # doesn't memorize exact pixel locations.
            transforms.RandomCrop(32, padding=4), 
            
            # 2. Standard flip
            transforms.RandomHorizontalFlip(), 
            
            # 3. Convert to Tensor
            transforms.ToTensor(), 
            
            # 4. Normalize
            transforms.Normalize(self.mean, self.std),
            
            # 5. RandomErasing (Cutout). Randomly blocks parts of the image.
            # This forces the model to look at the WHOLE object, not just one feature.
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.cifar10_t = datasets.CIFAR10(
            data_path, train=True, download=True,
            transform=self.train_transform
        )

        self.cifar10_val_t = datasets.CIFAR10(
            data_path, train=False, download=True,
            transform=self.val_transform
        )

    def get_datasets(self):
        return self.cifar10_t, self.cifar10_val_t