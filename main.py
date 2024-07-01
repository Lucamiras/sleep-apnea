from src.preprocess import Preprocessor
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
    ids_to_process=['00001342']
)

#pre.get_download_folder_contents()
pre.run(download=False)
#print(pre.label_dictionary)
#print(pre.segments_dictionary)
