from src.dataset import SignalDataset
from src.preprocess import Preprocessor
from src.preprocess import get_download_urls
from src.utils.imagedata import get_mean_and_std
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

run_pre = False

if run_pre:
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

INPUT_SIZE = (224, 224)

transform_without_normalization = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
    ])

train_without_norm = SignalDataset('data/processed/signals/train', transform=transform_without_normalization, classes=CLASSES, shuffle=False)

mean, std = get_mean_and_std(train_without_norm)
mean = list(mean.numpy())
std = list(std.numpy())

print(mean, std)


