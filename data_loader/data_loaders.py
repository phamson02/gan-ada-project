from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.custom_datasets import *

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, train_portion=1.0, num_workers=1, pin_memory=False,
                 drop_last=False, training=True):
        trsfm = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, train_portion, num_workers, pin_memory=pin_memory, drop_last=drop_last)

class CelebA64DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, img_size=64, train_portion=0.9, shuffle=True, num_workers=1, pin_memory=False, drop_last=False, training=True):
        trsfm = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.data_dir = data_dir
        self.dataset = CelebA64(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, train_portion, num_workers, pin_memory=pin_memory, drop_last=drop_last)

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, img_size=32, shuffle=True, train_portion=1.0, num_workers=1, pin_memory=False, drop_last=False, training=True):
        self.data_dir = data_dir
        trsfm = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.dataset = datasets.CIFAR10(self.data_dir, transform=trsfm, download=True)
        super().__init__(self.dataset, batch_size, shuffle, train_portion, num_workers, pin_memory=pin_memory, drop_last=drop_last)


class HighResolutionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, img_size=512, shuffle=True, train_portion=1.0, num_workers=1, pin_memory=False, drop_last=False, training=True):
        transform_list = [
            transforms.Resize((int(img_size), int(img_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        transf = transforms.Compose(transform_list)
        self.data_dir = data_dir
        self.dataset = FFHQ(self.data_dir, transform=transf)
        super().__init__(self.dataset, batch_size, shuffle, train_portion, num_workers, pin_memory=pin_memory,
                         drop_last=drop_last)
