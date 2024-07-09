from src.preprocess import Preprocessor
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)
import warnings

warnings.filterwarnings('ignore')

pre = Preprocessor(project_dir='data',
                   edf_urls=EDF_URLS,
                   rml_urls=RML_URLS,
                   data_channels=DATA_CHANNELS,
                   classes=CLASSES,
                   ids_to_process=['00000995'],
                   clip_length=20.0)

pre.run(download=False)
