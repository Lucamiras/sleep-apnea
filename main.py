import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import h5py
from torch.utils.data import DataLoader
from torchvision import transforms
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
from src.dataset import SignalDataset

# Initialize pipeline elements

overrides = {
    "ids_to_process":['00000995','00001006'],
    "clip_length":5,
    "new_sample_rate": 16_000,
    "augment_ratio":0.2
}

config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=True,
    process_signals=True,
    serialize_signals=True,
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

#pre.run()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
])
dataset = SignalDataset('data/processed/signals/dataset.h5', 'Train', transform=transform, classes=CLASSES)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

img, label = next(iter(dataloader))

plt.imshow(img.squeeze()[500])
plt.show()
