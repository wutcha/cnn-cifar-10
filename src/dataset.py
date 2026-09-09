import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.RandomResizedCrop(size=(32,32), scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15, 15))
    ])

test_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

train_data = datasets.CIFAR10(root = 'data', train = True, download = True, transform = train_transform)
test_data = datasets.CIFAR10(root = 'data', train = False, download = True, transform = test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)