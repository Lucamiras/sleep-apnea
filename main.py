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

pre._create_directory_structure()
pre._move_selected_downloads_to_preprocessing()
pre._create_label_dictionary()
pre._get_edf_segments_from_labels()
pre._create_wav_data()
pre._collect_processed_raw_files()
