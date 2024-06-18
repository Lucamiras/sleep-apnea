import os
import requests

class Preprocessor:
    def __init__(self,
                 project_dir: str,
                 edf_urls: list,
                 rml_urls: list,
                 data_channels: list,
                 classes: dict):
        self.project_dir = project_dir
        self.edf_urls = edf_urls
        self.rml_urls = rml_urls
        self.data_channels = data_channels
        self.classes = classes

    def _create_directory_structure(self) -> None:
        os.makedirs(os.path.join(self.project_dir, 'data', 'downloads', 'edfs'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'data', 'downloads', 'rmls'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'data', 'processed', 'parquet'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'data', 'downloads', 'audio'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'data', 'downloads', 'spectrogram'), exist_ok=True)
        
    def _download_data(self) -> None:
        edf_path = os.path.join(self.project_dir, 'data', 'downloads', 'edfs')
        rml_path = os.path.join(self.project_dir, 'data', 'downloads', 'rmls')

        for url in self.edf_urls:
            response = requests.get(url)
            file_name = url[-28:]
            file_path = os.path.join(edf_path, file_name).replace('%5B', '[').replace('%5D', ']')
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in self.rml_urls:
            response = requests.get(url)
            file_name = url[-19:]
            file_path = os.path.join(rml_path, file_name).replace('%5B', '[').replace('%5D', ']')
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print("Download of labels completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")