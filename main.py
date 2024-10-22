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
import h5py
import json
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
config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=False,
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
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
dataset = SignalDataset(
    'data/processed/signals/dataset.h5',
    transform=transform,
    classes=CLASSES
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for image, labels in loader:
    print(image.shape)
    print(labels)
