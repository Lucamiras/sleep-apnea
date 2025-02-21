import os
import json
import argparse
from src.preprocessing.steps.config import Config
from src.preprocessing.steps.download import Downloader
from src.preprocessing.steps.extract import Extractor
from src.preprocessing.steps.process import Processor
from src.preprocessing.steps.serializer import Serializer
from src.preprocessing.pipeline import DataPreprocessor
from src.config.apnea_classes import apnea_classes

def main(arguments):
    download = arguments.download
    extract = arguments.extract
    process = arguments.process
    serialize = arguments.serialize
    overrides = {}

    if not arguments.acq_numbers:
        raise Exception("No acq_numbers provided.")

    if not arguments.targets:
        raise Exception("No targets provided. Provide a minimum of two classes.")

    if not (download or extract or process or serialize):
        raise Exception("No preprocessing steps selected. Select at least one preprocessing step.")

    acq_numbers = [acq_num.strip() for acq_num in arguments.acq_numbers.split(',')]
    targets = set(target.strip() for target in arguments.targets.split(','))
    classes_dictionary = {value: i for i, value in enumerate(apnea_classes & targets)}

    with open(os.path.join('src', 'config', 'override_config.json'), 'rb') as overrides_config:
        overrides = json.loads(overrides_config.read())

    config = Config(
        classes=classes_dictionary,
        ids_to_process=acq_numbers,
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

    pre.run()

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
        '--acq_numbers',
        help='Add list of acq numbers for all patients to include, using the format \"\'00001234\', \'00002345\'\" (\" str, str \".'
    )
    parser.add_argument(
        '--targets',
        help='Provide a string of apnea types. Allowed types are NoApnea, Hypopnea, ObstructiveApnea, MixedApnea, CentralApnea.'
    )
    args = parser.parse_args()
    main(args)