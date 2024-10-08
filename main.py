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
config.ids_to_process = ['00000995']
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

# pre.run()

INPUT_SIZE = (224, 224)
transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])

dataset = SignalDataset(os.path.join(config.signals_path, 'train'), transform=transforms, classes=config.classes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for images, labels in dataloader:
    print(images.shape)
    print(labels.shape)
    break