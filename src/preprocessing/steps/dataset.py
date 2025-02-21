from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import transforms


class SignalDataset(Dataset):
    def __init__(self, dataset:list, classes:dict, mean:float, std:float):
        self.dataset = dataset
        self.classes = classes
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        one_hot_label = self._get_onehot_label_from_classes_dictionary(sample)
        img = self._transform_data(sample)
        return img, one_hot_label

    def _transform_data(self, sample):
        img = Image.fromarray(sample['data'].astype(np.uint8))
        img = transforms.ToTensor()(img)
        if self.mean and self.std:
            img = transforms.Normalize(self.mean, self.std)(img)
        return img

    def _get_onehot_label_from_classes_dictionary(self, sample):
        return self.classes[sample['label']]