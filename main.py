from src.preprocess import Preprocessor
from src.utils.globals import (
    EDF_URLS, RML_URLS, DATA_CHANNELS, CLASSES
)

pre = Preprocessor(
    project_dir='data',
    edf_urls=EDF_URLS,
    rml_urls=RML_URLS,
    data_channels=DATA_CHANNELS,
    classes=CLASSES
)

pre.run(download=False)
print(pre.label_dictionary)

counters = {
    "NoApnea": 0,
    "Hypopnea": 0,
    "ObstructiveApnea": 0,
    "MixedApnea": 0,
}

for item in pre.label_dictionary['00000995']:
    label, _, _ = item
    counters[label] += 1

print(counters)

