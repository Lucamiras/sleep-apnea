import os
import numpy as np
from torchvision import transforms
from typing_extensions import override
from torch import optim
import torch
from src.nets.train_eval_loop import train_epoch
from src.dataset import SignalDataset
from torch.utils.data import DataLoader
from src.utils.imagedata import get_mean_and_std
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as f
from src.preprocess import (
    Config,
    Downloader,
    Extractor,
    Processor,
    Serializer,
    DataPreprocessor,
    Augmenter
)
from src.utils.globals import (
    CLASSES
)

# Initialize pipeline elements
config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=True,
    process_signals=True,
    serialize_signals=True,
)
config.ids_to_process = ['00001006']
config.audio_features = ['mel_spectrogram']
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

#pre.run()

augmenter = Augmenter({
    "gaussian": 0.2,
    "ambient": 0.2
}, config)

augmenter.augment()