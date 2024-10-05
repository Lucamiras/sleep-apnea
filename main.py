import os
import numpy as np
from torchvision import transforms
from src.dataset import SignalDataset
from torch.utils.data import DataLoader
from src.utils.imagedata import get_mean_and_std
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
config = Config(classes=CLASSES, download_files=False, extract_signals=False)
downloader = Downloader(config)
extractor = Extractor(config)
processor = Processor(config)
serializer = Serializer(config)

# Generate download URLS and update config class
edf_urls, rml_urls, unique_ids = downloader.get_download_urls()
config.edf_urls = edf_urls
config.rml_urls = rml_urls

# Initialize full pipeline
pre = DataPreprocessor(
    downloader,
    extractor,
    processor,
    serializer,
    config)
