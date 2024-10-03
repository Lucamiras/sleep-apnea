from src.dataset import SignalDataset
from src.preprocess import Preprocessor
from src.preprocess import get_download_urls
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


pre = Preprocessor(
    project_dir='data',
    edf_urls=EDF_URLS,
    rml_urls=RML_URLS,
    data_channels=DATA_CHANNELS,
    classes=CLASSES,
    clip_length=30,
    sample_rate=48000,
)

pre.run(
    download=False,
    dictionary=False,
    segments=False,
    create_files=False,
    signals_dict=True,
)

print(len(pre.signal_dictionary))
print(set(key.split('_')[1] for key in list(pre.signal_dictionary.keys())))
