from src.preprocess import Preprocessor
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
from src.augment import SpecAugmentation
from src.dataset import SpectrogramDataset
import warnings
import os
import xml.dom.minidom

warnings.filterwarnings('ignore')

pre = Preprocessor(project_dir='data',
                   edf_urls=EDF_URLS,
                   rml_urls=RML_URLS,
                   data_channels=DATA_CHANNELS,
                   classes=CLASSES,
                   clip_length=20,
                   clip_step_size=5,
                   ids_to_process=['00000995', '00001006'],
                   sample_rate=48000)

pre.run(download=False,
        dictionary=True,
        segments=True,
        create_files=True,
        shuffle=True)
