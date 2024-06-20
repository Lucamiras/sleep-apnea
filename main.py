from src.preprocess_new import Preprocessor
from src.utils.file_urls import get_download_urls

edf_urls, rml_urls = get_download_urls()

pre = Preprocessor(
    project_dir='data',
    edf_urls=edf_urls,
    rml_urls=rml_urls,
    data_channels=['Mic'],
    classes={
        "NoApnea": 0,
        "Hypopnea": 1,
        "ObstructiveApnea": 2,
        "MixedApnea": 3
    }
)

pre.run()
print(pre.label_dictionary)