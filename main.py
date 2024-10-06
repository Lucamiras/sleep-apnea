import os
import numpy as np
from torchvision import transforms
from src.dataset import SignalDataset
from torch.utils.data import DataLoader
from src.utils.imagedata import get_mean_and_std
import matplotlib.pyplot as plt
from src.preprocess import (
    Config,
    Downloader,
    Extractor,
    Processor,
    Serializer,
    DataPreprocessor
)
from src.utils.globals import (
    CLASSES
)

# Initialize pipeline elements
config = Config(classes=CLASSES)
config.ids_to_process = ['00001006']
downloader = Downloader(config)
extractor = Extractor(config)
processor = Processor(config)
serializer = Serializer(config)

# Initialize full pipeline
pre = DataPreprocessor(
    downloader,
    extractor,
    processor,
    serializer,
    config)
