from torch.utils.data import Dataset
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, root1, root2, transform, mode='train'):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.human = glob.glob(root1)
        self.cartoon = glob.glob(root2)
        self.new_perm()
        assert len(self.human) == len(self.cartoon), "Make sure both datasets have the same size"
    
    def new_perm(self):
        self.randperm = torch.randperm(len(self.cartoon))[:len(self.human)]

    def __getitem__(self, index): 
        if(self.transform):
            item_A = self.transform(Image.open(self.human[index % len(self.human)]))
            item_B = self.transform(Image.open(self.cartoon[self.randperm[index]]))
            
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1: 
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.human), len(self.cartoon))