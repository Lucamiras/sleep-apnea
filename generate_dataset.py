import argparse
from argparse import ArgumentParser

from src.data_preprocessing.config import Config
from src.data_preprocessing.downloader import Downloader
from src.data_preprocessing.extractor import Extractor
from src.data_preprocessing.process import Processor
from src.data_preprocessing.serializer import Serializer
from src.data_preprocessing.pipeline import DataPreprocessor
from src.utils.globals import (
    CLASSES
)

def main(arguments):
    download = arguments.download
    extract = arguments.extract
    process = arguments.process
    serialize = arguments.serialize

    if not arguments.acq_numbers:
        raise Exception("No acq_numbers provided.")

    if not (download or extract or process or serialize):
        raise Exception("No preprocessing steps selected. Select at least one preprocessing step.")

    acq_numbers = arguments.acq_numbers.split(',')

    overrides = {
        "ids_to_process":acq_numbers,
        "clip_length":30,
        "new_sample_rate": 16_000
    }

    config = Config(
        classes=CLASSES,
        download_files=download,
        extract_signals=extract,
        process_signals=process,
        serialize_signals=serialize,
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

    #pre.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--download',
        action='store_true',
        help='Set this flag if the EDF and RML files require downloading.'
    )
    parser.add_argument(
        '-e',
        '--extract',
        action='store_true',
        help='Set this flag if you want to extract labelled signals.'
    )
    parser.add_argument(
        '-p',
        '--process',
        action='store_true',
        help='Set this flag if you want to process the labelled signals into spectrograms.'
    )
    parser.add_argument(
        '-s',
        '--serialize',
        action='store_true',
        help='Set this flag if you want to pickle the dataset.'
    )
    parser.add_argument(
        '-a',
        '--acq_numbers',
        help='Add list of acq numbers for all patients to include, using the format \"\'00001234\', \'00002345\'\" (\" str, str \".'
    )
    args = parser.parse_args()
    main(args)