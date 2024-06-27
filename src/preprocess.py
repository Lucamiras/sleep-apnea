import os
import shutil
import requests
import mne
import xml.dom.minidom
import numpy as np
from tqdm import tqdm
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import gc
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class Preprocessor:
    """
    This Preprocessor class takes a list of EDF download urls and RML download urls and prepares the files
    for training in the CNN. The following steps are executed:

        1. Creating a data subdirectory in the project folder

        2. Downloading the files and moving them into the subdirectories.

        3. Creating segments from timestamps provided in the rml files.

        4. Creating spectrogram files and randomly shuffling them into train, validation and test folders.

    There are TWO ways of using this preprocessor: download = False and download = True

    If you choose download = False when calling
        > Preprocessor().run(download=False)
    you should have EDF files to process in the folder project_dir > downloads > edf as well as rml. If the files
    are already in the folder, use the property patient_ids_to_process.

    If you choose download = True, the EDF and RML URL properties must contain valid download URLs for the PSG-Audio
    dataset.
    """
    def __init__(self,
                 project_dir: str,
                 edf_urls: list,
                 rml_urls: list,
                 data_channels: list,
                 classes: dict,
                 sample_rate: int = 48_000,
                 train_size: float = .8,
                 ids_to_ignore: list = None):

        self.project_dir = project_dir
        self.edf_urls = edf_urls
        self.rml_urls = rml_urls
        self.data_channels = data_channels
        self.classes = classes
        self.sample_rate = sample_rate
        self.train_size = train_size
        self.ids_to_ignore = ids_to_ignore
        self.patient_ids = []
        self.label_dictionary = {}
        self.segments_dictionary = {}

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        :return: None
        """

        logging.info('1 --- Creating directory structure ---')

        self.edf_path: str = os.path.join(self.project_dir, 'downloads', 'edf')
        self.rml_path: str = os.path.join(self.project_dir, 'downloads', 'rml')
        self.audio_path: str = os.path.join(self.project_dir, 'processed', 'audio')
        self.spectrogram_path: str = os.path.join(self.project_dir, 'processed', 'spectrogram')
        self.retired_path: str = os.path.join(self.project_dir, 'retired')

        os.makedirs(self.edf_path, exist_ok=True)
        os.makedirs(self.rml_path, exist_ok=True)
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.spectrogram_path, exist_ok=True)
        os.makedirs(self.retired_path, exist_ok=True)

    def _download_data(self) -> None:
        """
        Downloads the files specified in the edf and rml urls.
        :return: None
        """
        logging.info(f"2 --- Starting download of {len(self.edf_urls)} EDF files and {len(self.rml_urls)} RML files ---")

        for url in tqdm(self.edf_urls):
            response = requests.get(url)
            file_name = url[-28:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.edf_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in tqdm(self.rml_urls):
            response = requests.get(url)
            file_name = url[-19:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.rml_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Download of {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        if len(self.edf_urls) == len(os.listdir(self.edf_path)):
            print("Successfully downloaded EDF files.")

        if len(self.rml_urls) == len(os.listdir(self.rml_path)):
            print("Successfully downloaded RML files.")

    def _organize_downloads(self):
        """
        Moves files into folders by patient id.
        :return:
        """
        logging.info('3 --- Organizing downloaded files into folders ---')

        # This block checks if the files I want to process are already in a folder. If yes, it moves the files
        # into the correct position

        edf_folder_contents = [file for file in os.listdir(self.edf_path) if file.endswith('.edf')]
        rml_folder_contents = [file for file in os.listdir(self.rml_path) if file.endswith('.rml')]
        self.patient_ids = list(set([uid.split('-')[0] for uid in edf_folder_contents]))

        for patient_id in self.patient_ids:
            os.makedirs(os.path.join(self.edf_path, patient_id), exist_ok=True)
            os.makedirs(os.path.join(self.rml_path, patient_id), exist_ok=True)
            print(self.patient_ids)

        for edf_file in edf_folder_contents:
            src_path = os.path.join(self.edf_path, edf_file)
            dst_path = os.path.join(self.edf_path, edf_file.split('-')[0], edf_file)
            shutil.move(src=src_path, dst=dst_path)

        for rml_file in rml_folder_contents:
            src_path = os.path.join(self.rml_path, rml_file)
            dst_path = os.path.join(self.rml_path, rml_file.split('-')[0], rml_file)
            shutil.move(src=src_path, dst=dst_path)

    def _create_label_dictionary(self):
        logging.info('4 --- Creating label dictionaries ---')
        clip_length = 10

        for rml_folder in os.listdir(self.rml_path):
            for file in os.listdir(os.path.join(self.rml_path, rml_folder)):
                label_path = os.path.join(self.rml_path, rml_folder, file)
                domtree = xml.dom.minidom.parse(label_path)
                group = domtree.documentElement
                events = group.getElementsByTagName("Event")
                events_apnea = []
                non_events_apnea = []

                for event in events:
                    event_type = event.getAttribute('Type')
                    if event_type in self.classes.keys():
                        positive_example = (str(event_type),
                                            float(event.getAttribute('Start')),
                                            float(clip_length))
                        events_apnea.append(positive_example)

                for i in range(len(events_apnea) -1):
                    current_event = events_apnea[i]
                    next_event = events_apnea[i + 1]
                    lower_limit = current_event[1] + current_event[2]
                    upper_limit = next_event[1]
                    if upper_limit - lower_limit > float(clip_length):
                        middle_value = (lower_limit + upper_limit) / 2
                        negative_example = ('NoApnea',
                                            float(middle_value - (clip_length / 2)),
                                            float(clip_length))
                        non_events_apnea.append(negative_example)

                all_events = events_apnea + non_events_apnea
                all_events = sorted(all_events, key=lambda x: x[1])

                self.label_dictionary[rml_folder] = all_events

    def _read_out_single_edf_file(self, edf_folder: str) -> np.array:
        """
        Reads out all files from a single edf directory and concatenates the channel information
        in a numpy array.
        :returns: None
        """
        edf_files: list = sorted([os.path.join(edf_folder, file) for file in os.listdir(edf_folder)])

        full_readout: np.ndarray = np.array([])
        step_size: int = 10_000_000

        for edf_file in edf_files:
            print(f'Starting readout for file {edf_file}')
            raw_edf_file = mne.io.read_raw_edf(edf_file, preload=False)
            length_of_file = len(raw_edf_file)
            for sliding_index in tqdm(range(0, length_of_file, step_size)):
                raw_edf_chunk = raw_edf_file.get_data(picks=['Mic'],
                                                      start=sliding_index,
                                                      stop=min((sliding_index + step_size, length_of_file)))
                full_readout = np.append(full_readout, raw_edf_chunk)

        return full_readout

    def _get_edf_segments_from_labels(self) -> None:
        """
        Load edf files and create segments by timestamps.
        :return:
        """
        logging.info('5 --- Create segments ---')
        clip_length_seconds = 10

        for edf_folder in os.listdir(self.edf_path):
            print(f"Starting to create segments for user {edf_folder}")
            edf_folder_path = os.path.join(self.edf_path, edf_folder)
            edf_readout = self._read_out_single_edf_file(edf_folder_path)
            self.segments_dictionary[edf_folder] = []

            for label in self.label_dictionary[edf_folder]:
                label_desc = label[0]
                time_stamp = label[1]
                start_idx = int(time_stamp * self.sample_rate)
                end_idx = int((time_stamp + clip_length_seconds) * self.sample_rate)
                segment = edf_readout[start_idx:end_idx]
                if len(segment) > 0:
                    self.segments_dictionary[edf_folder].append((label_desc, segment))

            del edf_readout, segment
            gc.collect()

    def _save_spectrogram(self, wav_file_path, dest_path) -> None:
        """
        This function takes a single wav file and creates a de-noised spectrogram.
        """
        # y is the audio signal
        # sr is the sample rate
        y, sr = librosa.load(wav_file_path, sr=self.sample_rate)  # retrieve audio signal and sample rate
        y = y / np.max(np.abs(y))
        y_denoised = nr.reduce_noise(y, sr)
        y_amplified = y_denoised * 10
        d_matrix = librosa.stft(y_amplified)
        d_decibels = librosa.amplitude_to_db(np.abs(d_matrix), ref=np.max)

        plt.figure(figsize=(2, 2))
        librosa.display.specshow(d_decibels, sr=sr, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.savefig(dest_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _create_all_spectrogram_files(self) -> None:
        """
        Goes through all wav files in the data/processed/audio subfolder and calls the save_spectrogram
        function."""
        logging.info('7 --- Creating spectrogram files ---')

        for index, wav_file in tqdm(enumerate(os.listdir(self.audio_path))):
            wav_file_path = os.path.join(self.audio_path, wav_file)
            spec_file_name = wav_file.split('.wav')[0]
            dest_path = os.path.join(self.spectrogram_path, spec_file_name)
            self._save_spectrogram(wav_file_path, dest_path)

    def _train_val_test_split_spectrogram_files(self) -> None:
        """
        This function shuffles the spectrogram files randomly and assigns them to train, validation or test folders
        according to the ratio defined in train_val_test_ratio.
        :returns: None
        """
        logging.info('8 --- Splitting into train, validation and test ---')

        validation_size = (1 - self.train_size) / 2
        spectrogram_files = [file for file in os.listdir(self.spectrogram_path) if file.endswith('.png')]
        random.shuffle(spectrogram_files)
        total_number_of_files = len(spectrogram_files)

        train_index = int(total_number_of_files * self.train_size)  # 80
        validation_index = int(total_number_of_files * validation_size) + train_index

        train_files = spectrogram_files[:train_index]
        validation_files = spectrogram_files[train_index:validation_index]
        test_files = spectrogram_files[validation_index:]

        self._move_files(train_files, 'train')
        self._move_files(validation_files, 'validation')
        self._move_files(test_files, 'test')

    def _move_files(self, files, target_folder) -> None:
        """
        Move files to the target folder.
        """
        destination_path = os.path.join(self.spectrogram_path, target_folder)
        os.makedirs(destination_path, exist_ok=True)
        number_of_files = len(files)

        for file in files:
            source = os.path.join(self.spectrogram_path, file)
            destination = os.path.join(destination_path, file)
            shutil.move(src=source, dst=destination)

        print(f'Moved {number_of_files} files into /{target_folder}.')

    def _create_wav_data(self, pcm_rate: int = 32768) -> None:
        """
        This function iterates through the segments list and creates individual wav files.
        :returns: None
        """
        logging.info('6 --- Creating WAV files ---')

        for edf_folder in os.listdir(self.edf_path):
            for index, annotated_segment in enumerate(self.segments_dictionary[edf_folder]):
                label, signal = annotated_segment
                class_index = self.classes[label]
                file_name_wav = f'{index:05d}_{edf_folder}_{label}_{class_index}.wav'

                audio_data = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
                audio_data_pcm = np.int16(audio_data * pcm_rate)
                wav_path = os.path.join(self.audio_path, file_name_wav)
                write(wav_path, self.sample_rate, audio_data_pcm)

    def _remove_ignored_files(self) -> None:
        # loop through edf
        retired_edf_path = os.path.join(self.retired_path, 'edf')
        retired_rml_path = os.path.join(self.retired_path, 'rml')

        os.makedirs(retired_edf_path, exist_ok=True)
        os.makedirs(retired_rml_path, exist_ok=True)

        for edf_folder in self.ids_to_ignore:
            if edf_folder in os.listdir(self.edf_path):
                shutil.move(os.path.join(self.edf_path, edf_folder), os.path.join(retired_edf_path, edf_folder))

        for rml_folder in self.ids_to_ignore:
            if rml_folder in os.listdir(self.rml_path):
                shutil.move(os.path.join(self.rml_path, rml_folder), os.path.join(retired_rml_path, rml_folder))

    def run(self, download: bool = True) -> None:
        """os.makedirs(self.parquet_path, exist_ok=True)
        Runs the preprocessing pipeline.
        :return: None
        """
        self._create_directory_structure()
        if self.ids_to_ignore is not None:
            self._remove_ignored_files()
        if download:
            self._download_data()
        self._organize_downloads()
        self._create_label_dictionary()
        self._get_edf_segments_from_labels()
        self._create_wav_data()
        self._create_all_spectrogram_files()
        self._train_val_test_split_spectrogram_files()
