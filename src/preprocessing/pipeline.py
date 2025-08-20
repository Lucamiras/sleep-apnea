import os
from src.preprocessing.steps.config import Config
from src.preprocessing.steps.download import LocalDownloader, GCSDownloader
from src.preprocessing.steps.extract import Extractor
from src.preprocessing.steps.process import Processor


class DataPreprocessor:
    def __init__(self,
                 config: Config):
        self.config = config
        self.downloader = None
        self.extractor = None
        self.processor = None

    def run(self):
        # Create directories if not already present
        self._create_directory_structure()

        if "download" in self.config.steps:
            # This will download any files specified in the EDF_URLS and RML_URLS from the Config class
            if self.config.download.mode == "local":
                self.downloader = LocalDownloader(self.config)
            if self.config.download.mode == "gcs":
                self.downloader = GCSDownloader(self.config)
            self.downloader.download_data()

        if "extract" in self.config.steps:
            # This will extract audio signals from the EDF files according to the RML files and save them as NPZ files.
            self.extractor = Extractor(self.config)
            self.extractor.extract()

        if "process" in self.config.steps:
            # This will process the extracted signals, and process it into audio features
            self.processor = Processor(self.config)
            self.processor.process()

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        returns: None
        """
        os.makedirs(self.config.paths.edf_download_path, exist_ok=True)
        os.makedirs(self.config.paths.rml_download_path, exist_ok=True)
        os.makedirs(self.config.paths.edf_preprocess_path, exist_ok=True)
        os.makedirs(self.config.paths.rml_preprocess_path, exist_ok=True)
        os.makedirs(self.config.paths.pickle_path, exist_ok=True)
        os.makedirs(self.config.paths.audio_path, exist_ok=True)
        os.makedirs(self.config.paths.spectrogram_path, exist_ok=True)
        os.makedirs(self.config.paths.signals_path, exist_ok=True)
        os.makedirs(self.config.paths.retired_path, exist_ok=True)