import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# class ApneaDataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform
# 
#     def __getitem__(self, index):
#         image = Image.fromarray(self.data[index]).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image
#     
#     def __len__(self): 
#         return len(self.data)


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as f
from PIL import Image
from tqdm import tqdm
import os


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None):
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.spectrogram_files = sorted([file for file in os.listdir(spectrogram_dir)])

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.spectrogram_dir, self.spectrogram_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
