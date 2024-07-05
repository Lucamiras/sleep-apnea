from src.preprocess import Preprocessor
from src.dataset import SpectrogramDataset
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os

warnings.filterwarnings('ignore')

pre = Preprocessor(
    project_dir='data',
    edf_urls=EDF_URLS,
    rml_urls=RML_URLS,
    data_channels=DATA_CHANNELS,
    classes=CLASSES,
    ids_to_process=['00000995']
)

#pre.run(download=False)
dataset = SpectrogramDataset('data/processed/spectrogram/train', classes=CLASSES, transform=None)
print(dataset.get_mean_std())


