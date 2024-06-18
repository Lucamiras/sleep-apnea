import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import os


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None):
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.spectrogram_files = [file for file in os.listdir(spectrogram_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.spectrogram_dir, self.spectrogram_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self._get_label_from_filename(self.spectrogram_files[idx])
        return image, label

    def _get_label_from_filename(self, file_name):
        return int(file_name.split('.png')[0][-1])