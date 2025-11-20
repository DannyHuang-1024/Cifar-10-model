import torch
from torchvision import transforms
from torchvision import datasets



class CfDatasets:
    def __init__(self):
        data_path = r"~/data"
        self.mean, self.std = self.calculate_args(data_path)

        self.cifar10_t = datasets.CIFAR10(
            data_path, train=True, download=False,
            transform=transforms.Compose(
                [transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(), # Flip the img horizontally
                transforms.RandomRotation(15), # Rotate the img for a certain angle
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                ]
                # img = (r*, g*, b*), mean/std = (r, g, b)
            )
        )

        self.cifar10_val_t = datasets.CIFAR10(
            data_path, train=False, download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]
                # img = (r*, g*, b*), mean/std = (r, g, b)
            )
        )

    @staticmethod
    def calculate_args(path):
        cifar10_t = datasets.CIFAR10(
            path, train=True, download=False,
            transform = transforms.ToTensor()
        )

        temp = torch.stack([img_t for img_t, _ in cifar10_t], dim=3) # Reshape as (3, XXXX)
        # print(temp.shape)
        return temp.view(3,-1).mean(dim = 1), temp.view(3,-1).std(dim=1)

    def get_datasets(self):
        cifar = [(img, label) for img, label in self.cifar10_t]
        cifar_val = [(img, label) for img, label in self.cifar10_val_t]
        
        return cifar, cifar_val

    



    