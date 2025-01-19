import os
import pickle
import shutil
from src.dataclasses.spectrogram import Spectrogram, SpectrogramDataset
from src.data_preprocessing.config import Config


class Serializer:
    """
    A class responsible for serializing data.
    """
    def __init__(self, config: Config, dataset: SpectrogramDataset):
        self.config = config
        self.dataset = dataset

    def serialize(self):
        dataset_file_path = os.path.join(self.config.signals_path, self.config.dataset_file_name)
        retired_file_path = os.path.join(self.config.retired_path, self.config.dataset_file_name)
        if self.config.dataset_file_name in os.listdir(self.config.signals_path):
            shutil.move(src=dataset_file_path, dst=retired_file_path)
        with open(dataset_file_path, "wb") as file:
            pickle.dump(self.dataset, file)