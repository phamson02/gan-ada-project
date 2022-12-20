import torch
import numpy as np
from torch.utils.data import Dataset
import glob as glob
import os
from PIL import Image

class CelebA64(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        self.len = len(self.img_paths)
        self.img_paths = self.img_paths[:self.len]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # print("hello", len(self.img_paths), index)
        img = Image.open(self.img_paths[index])

        if self.transform:
            img = self.transform(img)
        return (img, 1)

class FFHQ(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(self.data_dir, "*.png"))
        self.len = len(self.img_paths)
        self.img_paths = self.img_paths[:self.len]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])

        if self.transform:
            img = self.transform(img)
        return (img, 1)