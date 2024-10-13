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
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])

dataset = SignalDataset(os.path.join(config.signals_path, 'train'), transform=transform, classes=config.classes)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

mean, std = None, None

for images, labels in dataloader:
    batch_size = images.size(0)
    images = images.view(batch_size, images.size(1), -1)
    mean = images.mean(dim=(0,2))
    std = images.std(dim=(0,2))

transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1,0.1], std=[0.1,0.1]),
    ])

train_dataset = SignalDataset(os.path.join(config.signals_path, 'train'), transform=transform, classes=CLASSES)
val_dataset = SignalDataset(os.path.join(config.signals_path, 'val'), transform=transform, classes=CLASSES)
test_dataset = SignalDataset(os.path.join(config.signals_path, 'test'), transform=transform, classes=CLASSES)

generator1 = torch.Generator().manual_seed(42)
generator2 = torch.Generator().manual_seed(42)
generator3 = torch.Generator().manual_seed(42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    print(images.shape)
    image = images[0][0]
    print(image.shape)
    plt.imshow(image, cmap='inferno')
    plt.show()
    break