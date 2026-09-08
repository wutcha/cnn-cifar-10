import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

train_data = datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)