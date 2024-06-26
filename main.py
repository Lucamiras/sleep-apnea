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

#pre.get_download_folder_contents()
pre.run(download=False)
#pre.get_train_val_test_distributions()
#print(pre.label_dictionary)
#print(pre.segments_dictionary)
