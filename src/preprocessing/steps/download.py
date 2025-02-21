import os
import requests
import numpy as np
from tqdm import tqdm
import re


class Downloader:
    """
    A class responsible for downloading EDF and RML files from specified URLs.

    Attributes:
        config (Config): An instance of the Config class containing download settings and file paths.

    Methods:
        download_data():
            Downloads files from URLs specified in the config.
    """
    def __init__(self, config):
        """
        Initializes the Downloader with the given configuration.

        Args:
            config (Config): Configuration object containing paths and download settings.
        """
        self.config = config

    def get_download_urls(self, n_ids: int = 5, seed=42) -> tuple:
        """
            This function generated the EDF and matching RML URLs needed for the preprocessor.
            How to use: Download the data catalog file for the PSG Audio dataset and put it somewhere in your file
            directory. Specify the catalog file path when calling the function.
            :returns: A tuple with two lists (RML and EDF URLs) as well as a list of unique IDs.
        """
        np.random.seed(seed)
        with open(self.config.catalog_filepath, 'r') as f:
            content = f.read()
        urls = content.split('\n')
        ids = list(x.split('/')[1] for x in set(re.findall(pattern=r'clean/0000[0-9]{4}', string=content)))
        selected_ids = np.random.choice(a=ids, size=n_ids, replace=True).tolist()
        selected_rml_urls = []
        selected_edf_urls = []
        for s_id in selected_ids:
            for url in urls:
                if ('/V3/APNEA_RML_clean/' + s_id in url) and (url.endswith('.rml')):
                    selected_rml_urls.append(url)
                if ('/V3/APNEA_EDF/' + s_id in url) and (url.endswith('.edf')):
                    selected_edf_urls.append(url)
        return selected_rml_urls, selected_edf_urls, selected_ids

    def download_data(self) -> None:
        """
        Downloads files from URLs specified in the config.

        Returns:
            None
        """
        for url in tqdm(self.config.edf_urls):
            response = requests.get(url)
            file_name = url[-28:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.config.edf_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in tqdm(self.config.rml_urls):
            response = requests.get(url)
            file_name = url[-19:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.config.rml_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Download of {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        if len(self.config.edf_urls) == len(os.listdir(self.config.edf_download_path)):
            print("Successfully downloaded EDF files.")

        if len(self.config.rml_urls) == len(os.listdir(self.config.rml_download_path)):
            print("Successfully downloaded RML files.")
