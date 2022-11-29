from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.custom_datasets import *

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False,
                 drop_last=False, training=True):
        trsfm = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory, drop_last=drop_last)

class CelebA64DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, img_size=64, train_portion=0.9, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False, drop_last=False, training=True):
        trsfm = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.data_dir = data_dir
        self.dataset = CelebA64(self.data_dir, transforms=trsfm, train_portion=train_portion)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory, drop_last=drop_last)

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, img_size=32, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False, drop_last=False, training=True):
        self.data_dir = data_dir
        trsfm = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.dataset = datasets.CIFAR10(self.data_dir, transform=trsfm, download=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory=pin_memory, drop_last=drop_last)
