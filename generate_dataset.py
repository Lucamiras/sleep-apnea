import argparse
from src.data_preprocessing.config import Config
from src.data_preprocessing.downloader import Downloader
from src.data_preprocessing.extractor import Extractor
from src.data_preprocessing.process import Processor
from src.data_preprocessing.serializer import Serializer
from src.data_preprocessing.pipeline import DataPreprocessor
from src.utils.globals import (
    CLASSES
)


overrides = {
    "ids_to_process":['00000995'],
    "clip_length":30,
    "new_sample_rate": 16_000
}

config = Config(
    classes=CLASSES,
    download_files=False,
    extract_signals=True,
    process_signals=True,
    serialize_signals=True,
    overrides=overrides,
)
downloader = Downloader(config)
extractor = Extractor(config)
processor = Processor(config)
serializer = Serializer(config, processor.spectrogram_dataset)

pre = DataPreprocessor(
    downloader,
    extractor,
    processor,
    serializer,
    config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--download',
        action='store_true',
        help='Set this flag if the EDF and RML files require downloading.'
    )
    args = parser.parse_args()
    pre.run()