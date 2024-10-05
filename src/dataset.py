import torch
from torch.utils.data import Dataset
import torch.nn.functional as f
from PIL import Image
import os
import pickle
import random
import numpy as np


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None, classes: dict = None):
        assert classes is not None, "No classes were selected. No data will be loaded."
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.classes = classes
        self.num_classes = len(set(classes.values()))
        self.class_names = [key for key, value in self.classes.items()]
        self.spectrogram_files = [file for file in os.listdir(spectrogram_dir)
                                  if file.endswith('.png')
                                  if file.split('.png')[0].split('_')[2] in self.class_names]
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

    def _get_label_from_filename(self, file_name) -> list:
            """
            Extracts the label from the given file name.

            Args:
                file_name (str): The name of the file.

            Returns:
                list: The label corresponding to the file name.
            """
            return self.classes[file_name.split('.png')[0].split('_')[2]]

class SignalDataset(Dataset):
    def __init__(self, signal_dir, transform=None, classes:dict = None):
        assert classes is not None, "No classes were selected. Not data will be loaded."
        self.signal_dir = signal_dir
        self.transform = transform
        self.classes = classes
        self.signals = self._load_data_from_pickle_and_transform()
        self.num_classes = (len(set(classes.values())))
        self.patient_ids = set([patient_id.split('_')[1] for patient_id in self.signals.keys()])

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal_filename, signal_data = list(self.signals.items())[idx]
        signal_label = self._get_label_from_filename(signal_filename)
        image = Image.fromarray(signal_data.astype(np.uint8))
        one_hot_label = f.one_hot(torch.tensor(signal_label), num_classes=self.num_classes)
        if self.transform:
            signal_data = self.transform(image)
        return signal_data, one_hot_label

    def _get_label_from_filename(self, file_name):
        return self.classes[file_name.split('_')[2]]

    def _load_data_from_pickle_and_transform(self):
        # load from pickle
        pickle_files = os.listdir(self.signal_dir)
        all_signals = {}
        for file in pickle_files:
            with open(os.path.join(self.signal_dir, file), 'rb') as content:
                signals = pickle.load(content)
                all_signals.update(signals)
        signal_items = list(all_signals.items())
        all_signals = {key: value for key, value in signal_items}
        return all_signals
