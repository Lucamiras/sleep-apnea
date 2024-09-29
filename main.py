from src.preprocess import Preprocessor
from src.preprocess import get_download_urls
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings


warnings.filterwarnings('ignore')

pre = Preprocessor(
    project_dir='data',
    edf_urls=EDF_URLS,
    rml_urls=RML_URLS,
    data_channels=DATA_CHANNELS,
    classes=CLASSES,
    clip_length=30,
    sample_rate=48000,
    ids_to_process=['00000995']
)

pre.run(
    download=False,
    dictionary=True,
    segments=True,
    create_files=False,
    mel_frequencies_dict=True,
    shuffle=False
)

print(pre.mel_frequency_dictionary)
