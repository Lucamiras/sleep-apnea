import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as f
from PIL import Image
import numpy as np
import h5py


class SignalDataset(Dataset):
    def __init__(self, dataset_filepath, dataset:str='Train', transform=None, classes:dict = None, resize_to:tuple=(224,224)):
        assert classes is not None, "No classes were selected. Not data will be loaded."
        assert dataset in ['Train','Val','Test'], "Dataset must be Train, Val, or Test."
        self.dataset_filepath = dataset_filepath
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.resize_to = resize_to
        self.class_names = [key for key, value in self.classes.items()]
        self.signals = self._load_data_from_hdf5()
        self.num_classes = (len(set(classes.values())))
        self.patient_ids = set([patient_id.split('_')[1] for patient_id in self.signals.keys()])

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal_filename, signal_data = list(self.signals.items())[idx]
        signal_label = self._get_label_from_filename(signal_filename)
        one_hot_label = f.one_hot(torch.tensor(signal_label), num_classes=self.num_classes)
        num_features = signal_data.shape[0]
        feature_array = []
        for i in range(num_features):
            data = signal_data[i]
            data = np.where(np.isfinite(data), data, 0)
            data = self._rescale_array(data)
            data = Image.fromarray(data)
            if self.transform:
                data = self.transform(data)
            feature_array.append(data.squeeze())
        signal_data = torch.stack(feature_array)
        return signal_data, one_hot_label

    def _get_label_from_filename(self, file_name):
        return self.classes[file_name.split('_')[-1]]

    def _load_data_from_hdf5(self):
        dataset = {}
        with h5py.File(self.dataset_filepath, 'r') as hdf:
            for key, item in hdf[self.dataset].items():
                dataset[key] = item[()]
        all_signals = {key: value for key, value in dataset.items()
                       if key.split('_')[-1] in self.class_names}
        return all_signals

    @staticmethod
    def _rescale_array(x):
        x_min, x_max = np.min(x), np.max(x)
        return ((x - x_min) / (x_max - x_min) * 255).astype(np.uint8)