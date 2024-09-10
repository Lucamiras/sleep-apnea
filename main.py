from src.preprocess import Preprocessor
from src.preprocess import get_download_urls
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings
import os

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

rml, edf = get_download_urls(file_path=os.path.join('data', 'file_catalog.txt'), n_ids=2, seed=42)
print(rml)
print(edf)
