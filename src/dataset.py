import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as f
from PIL import Image
import os


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None, num_classes=None):
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.spectrogram_files = [file for file in os.listdir(spectrogram_dir) if file.endswith('.png')]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.spectrogram_dir, self.spectrogram_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self._get_label_from_filename(self.spectrogram_files[idx])
        one_hot_label = f.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return image, one_hot_label


    def _get_label_from_filename(self, file_name):
        return int(file_name.split('.png')[0][-1])