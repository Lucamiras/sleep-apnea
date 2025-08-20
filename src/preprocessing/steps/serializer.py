import os
import pickle
import shutil
from src.preprocessing.steps.config import Config


class Serializer:
    """
    A class responsible for serializing data.
    """
    def __init__(self, config: Config, dataset: dict):
        self.config = config
        self.dataset = dataset

    def serialize(self) -> None:
        dataset_file_path = os.path.join(self.config.paths.signals_path, self.config.paths.dataset_file_name)
        retired_file_path = os.path.join(self.config.paths.retired_path, self.config.paths.dataset_file_name)
        if self.config.paths.dataset_file_name in os.listdir(self.config.paths.signals_path):
            shutil.move(src=dataset_file_path, dst=retired_file_path)
        with open(dataset_file_path, "wb") as file:
            pickle.dump(self.dataset, file)