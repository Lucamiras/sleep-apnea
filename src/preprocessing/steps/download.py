from google.cloud import storage
import logging
import numpy as np
import os
import re
import requests
from tqdm import tqdm
from src.preprocessing.core.download import Downloader
from src.preprocessing.steps.config import Config


logging.basicConfig(level=logging.INFO)

class GCSDownloader(Downloader):
    def __init__(self, config:Config):
        self.config = config
        self.storage_client = storage.Client()

    def download_data(self):
        try:
            bucket = self.storage_client.bucket(self.config.download.bucket_name)
            for edf_url in tqdm(self.config.download.edf_urls):
                logging.info(f"Beginning download of {edf_url}")
                destination_blob_name = edf_url[-28:].replace('%5B', '[').replace('%5D', ']')
                response = requests.get(edf_url, stream=True)
                response.raise_for_status()
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_file(response.raw, content_type=response.headers['Content-Type'])
                logging.info(f"Successfully uploaded to gs://{self.config.download.bucket_name}/{destination_blob_name}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading file: {e}")
            raise
        except Exception as e:
            logging.error(f"Error downloading file: {e}")

#    try:
#        # 1. Download the file from the source URL
#        print(f"Downloading file from {source_url}...")
#        response = requests.get(source_url, stream=True)
#        response.raise_for_status()  # Raise an exception for bad status codes
#
#        # 2. Get the GCS bucket and create a blob object
#        bucket = self.storage_client.bucket(self.bucket_name)
#        blob = bucket.blob(destination_blob_name)
#
#        # 3. Upload the downloaded content directly from the stream
#        # This avoids saving the file to local disk first
#        blob.upload_from_file(response.raw, content_type=response.headers['Content-Type'])
#
#        print(f"Successfully uploaded to gs://{self.bucket_name}/{destination_blob_name}")
#
#    except requests.exceptions.RequestException as e:
#        print(f"Error downloading file: {e}")
#        raise
#    except Exception as e:
#        print(f"An error occurred: {e}")
#        raise


class LocalDownloader(Downloader):
    """
    A class responsible for downloading EDF and RML files from specified URLs.

    Attributes:
        config (Config): An instance of the Config class containing download settings and file paths.

    Methods:
        download_data():
            Downloads files from URLs specified in the config.
    """
    def __init__(self, config:Config):
        """
        Initializes the LocalDownloader with the given configuration.

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
        catalog_file_path = os.path.join(self.config.paths.root, self.config.download.catalog_file)
        with open(catalog_file_path, 'r') as f:
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
        for url in tqdm(self.config.download.edf_urls):
            response = requests.get(url)
            file_name = url[-28:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.config.paths.edf_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in tqdm(self.config.download.rml_urls):
            response = requests.get(url)
            file_name = url[-19:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.config.paths.rml_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Download of {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        if len(self.config.download.edf_urls) == len(os.listdir(self.config.paths.edf_download_path)):
            print("Successfully downloaded EDF files.")

        if len(self.config.download.rml_urls) == len(os.listdir(self.config.paths.rml_download_path)):
            print("Successfully downloaded RML files.")
