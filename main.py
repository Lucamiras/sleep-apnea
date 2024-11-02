import os
import numpy as np
from torchvision import transforms
import librosa
from PIL import Image
from typing_extensions import override
from torch import optim
import torch
from src.nets.train_eval_loop import train_epoch, eval_model
from src.nets.basicCNN import CNN
from src.dataset import SignalDataset
import soundfile as sf
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
    DataPreprocessor,
)
from src.utils.globals import (
    CLASSES
)

# Initialize pipeline elements

overrides = {
    "ids_to_process":['00000995'],
    "augment_ratio":0.5,
}

config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=True,
    process_signals=False,
    serialize_signals=False,
    overrides=overrides,
)
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

