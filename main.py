
from src.preprocess import Config, Downloader, DataPreprocessor
from src.utils.globals import (
    CLASSES
)

config = Config('data', CLASSES, download_files=True)
downloader = Downloader(config)
pre = DataPreprocessor(downloader, config)