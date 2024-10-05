
from src.preprocess import Config, Downloader, Extractor, Processor, Serializer, DataPreprocessor
from src.utils.globals import (
    CLASSES
)

config = Config(project_dir='data', classes=CLASSES, download_files=False, extract_signals=False)

pre = DataPreprocessor(
    downloader = Downloader(config),
    extractor = Extractor(config),
    processor = Processor(config),
    serializer = Serializer(config),
    config = config)

pre.run()