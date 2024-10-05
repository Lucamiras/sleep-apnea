from src.preprocess import (
    Config,
    Downloader,
    Extractor,
    Processor,
    Serializer,
    DataPreprocessor
)

from src.utils.globals import (
    CLASSES
)

config = Config(project_dir='data', classes=CLASSES, download_files=False, extract_signals=False)
downloader = Downloader(config)
extractor = Extractor(config)
processor = Processor(config)
serializer = Serializer(config)

edf_urls, rml_urls, unique_ids = downloader.get_download_urls()

config.edf_urls = edf_urls
config.rml_urls = rml_urls

pre = DataPreprocessor(
    downloader,
    extractor,
    processor,
    serializer,
    config)

print(pre.config.edf_urls)