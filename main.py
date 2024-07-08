from src.preprocess import Preprocessor
from src.dataset import SpectrogramDataset
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os
from torchvision import transforms
from src.utils.imagedata import get_mean_and_std
import librosa

warnings.filterwarnings('ignore')

pre = Preprocessor(project_dir='data', edf_urls=EDF_URLS, rml_urls=RML_URLS, data_channels=DATA_CHANNELS, classes=CLASSES, ids_to_process=['00000995'])
