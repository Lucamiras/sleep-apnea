import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as f
from PIL import Image
import os


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None, classes: dict = None):
        assert classes is not None, "No classes were selected. No data will be loaded."
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.classes = classes
        self.num_classes = len(classes)
        self.class_names = [key for key, value in self.classes.items()]
        self.spectrogram_files = [file for file in os.listdir(spectrogram_dir)
                                  if file.endswith('.png')
                                  if file.split('_')[2] in self.class_names]
        self.patient_ids = set([file.split('_')[1] for file in self.spectrogram_files])
        print(f"Dataset initialized with these classes: {self.class_names}")

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

