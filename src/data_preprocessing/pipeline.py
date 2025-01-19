from src.data_preprocessing.config import Config
from src.data_preprocessing.downloader import Downloader
from src.data_preprocessing.extractor import Extractor
from src.data_preprocessing.process import Processor
from src.data_preprocessing.serializer import Serializer


class DataPreprocessor:
    def __init__(self,
                 downloader: Downloader,
                 extractor: Extractor,
                 processor: Processor,
                 serializer: Serializer,
                 config: Config):
        self.downloader = downloader
        self.extractor = extractor
        self.processor = processor
        self.serializer = serializer
        self.config = config

    def run(self):
        if self.config.download_files:
            # This will download any files specified in the EDF_URLS and RML_URLS from the Config class
            self.downloader.download_data()

        if self.config.extract_signals:
            # This will extract audio signals from the EDF files according to the RML files and save them as NPZ files.
            self.extractor.extract()

        if self.config.process_signals:
            # This will process the extracted signals, and process it into audio features
            self.processor.process()

        if self.config.serialize_signals:
            # This will take the processor signals and turn them into shuffled pickle files
            self.serializer.serialize()