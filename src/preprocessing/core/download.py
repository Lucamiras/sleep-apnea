from abc import ABC, abstractmethod


class Downloader(ABC):
    @abstractmethod
    def download_data(self):
        pass