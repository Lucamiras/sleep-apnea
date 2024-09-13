import os
import shutil
import requests
import xml.dom.minidom
import numpy as np
from tqdm import tqdm
import librosa
import librosa.feature
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import gc
import random
import logging
import pyedflib
import re

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
                 edf_step_size: int = 10_000_000,
                 sample_rate: int = 48_000,
                 clip_length: int = 30,
                 train_size: float = .8,
                 ids_to_process: list = None):

        self.project_dir = project_dir
        self.edf_urls = edf_urls
        self.rml_urls = rml_urls
        self.data_channels = data_channels
        self.classes = classes
        self.edf_step_size = edf_step_size
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.train_size = train_size
        self.ids_to_process = ids_to_process
        self.patient_ids = []
        self.label_dictionary = {}
        self.segments_dictionary = {}

        self._create_directory_structure()

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        :returns: None
        """

        logging.info('1 --- Creating directory structure ---')

        self.edf_download_path: str = os.path.join(self.project_dir, 'downloads', 'edf')
        self.rml_download_path: str = os.path.join(self.project_dir, 'downloads', 'rml')
        self.edf_preprocess_path: str = os.path.join(self.project_dir, 'preprocess', 'edf')
        self.rml_preprocess_path: str = os.path.join(self.project_dir, 'preprocess', 'rml')
        self.npz_path: str = os.path.join(self.project_dir, 'preprocess', 'npz')
        self.audio_path: str = os.path.join(self.project_dir, 'processed', 'audio')
        self.spectrogram_path: str = os.path.join(self.project_dir, 'processed', 'spectrogram')
        self.retired_path: str = os.path.join(self.project_dir, 'retired')

        os.makedirs(self.edf_download_path, exist_ok=True)
        os.makedirs(self.rml_download_path, exist_ok=True)
        os.makedirs(self.edf_preprocess_path, exist_ok=True)
        os.makedirs(self.rml_preprocess_path, exist_ok=True)
        os.makedirs(self.npz_path, exist_ok=True)
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.spectrogram_path, exist_ok=True)
        os.makedirs(self.retired_path, exist_ok=True)

    def _download_data(self) -> None:
        """
        Downloads the files specified in the edf and rml urls.
        :returns: None
        """
        logging.info(
            f"2 --- Starting download of {len(self.edf_urls)} EDF files and {len(self.rml_urls)} RML files ---")

        for url in tqdm(self.edf_urls):
            response = requests.get(url)
            file_name = url[-28:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.edf_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in tqdm(self.rml_urls):
            response = requests.get(url)
            file_name = url[-19:].replace('%5B', '[').replace('%5D', ']')
            file_path = os.path.join(self.rml_download_path, file_name)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Download of {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        if len(self.edf_urls) == len(os.listdir(self.edf_download_path)):
            print("Successfully downloaded EDF files.")

        if len(self.rml_urls) == len(os.listdir(self.rml_download_path)):
            print("Successfully downloaded RML files.")

    def _move_selected_downloads_to_preprocessing(self):
        """
        Moves files into folders by patient id.
        :returns:
        """
        logging.info('3 --- Organizing downloaded files into folders ---')

        # This block checks if the files I want to process are already in a folder. If yes, it moves the files
        # into the correct position

        edf_folder_contents = [file for file in os.listdir(self.edf_download_path) if file.endswith('.edf')]
        rml_folder_contents = [file for file in os.listdir(self.rml_download_path) if file.endswith('.rml')]

        if self.ids_to_process is not None:
            edf_folder_contents = [file for file in edf_folder_contents if file.split('-')[0] in self.ids_to_process]
            rml_folder_contents = [file for file in rml_folder_contents if file.split('-')[0] in self.ids_to_process]

        unique_edf_file_ids = set([file.split('-')[0] for file in edf_folder_contents])
        unique_rml_file_ids = set([file.split('-')[0] for file in rml_folder_contents])

        assert unique_edf_file_ids == unique_rml_file_ids, ("Some EDF or RML files don't have matching pairs. "
                                                            "Preprocessing will not be possible:"
                                                            f"{unique_edf_file_ids}, {unique_rml_file_ids}")

        print(f"Preprocessing the following IDs: {unique_edf_file_ids}")

        for unique_edf_file_id in unique_edf_file_ids:
            os.makedirs(os.path.join(self.edf_preprocess_path, unique_edf_file_id), exist_ok=True)

        for unique_rml_file_id in unique_rml_file_ids:
            os.makedirs(os.path.join(self.rml_preprocess_path, unique_rml_file_id), exist_ok=True)

        for edf_file in edf_folder_contents:
            src_path = os.path.join(self.edf_download_path, edf_file)
            dst_path = os.path.join(self.edf_preprocess_path, edf_file.split('-')[0], edf_file)
            shutil.move(src=src_path, dst=dst_path)

        for rml_file in rml_folder_contents:
            src_path = os.path.join(self.rml_download_path, rml_file)
            dst_path = os.path.join(self.rml_preprocess_path, rml_file.split('-')[0], rml_file)
            shutil.move(src=src_path, dst=dst_path)

    def _create_sequential_label_dictionary(self, rml_folder) -> None:
        """
        This function goes through the EDF file in chunks of a determined size, i.e. 30 seconds,
        and labels each chunk according to the provided RML file.
        """
        for file in os.listdir(os.path.join(self.rml_preprocess_path, rml_folder)):
            label_path = os.path.join(self.rml_preprocess_path, rml_folder, file)
            domtree = xml.dom.minidom.parse(label_path)
            group = domtree.documentElement
            events = group.getElementsByTagName('Event')
            last_event_timestamp = int(float(events[-1].getAttribute('Start')))
            segment_duration = self.clip_length
            events_timestamps = [
                (float(event.getAttribute('Start')),
                 float(event.getAttribute('Duration')),
                 event.getAttribute('Type')) for event in events
                if event.getAttribute('Type') in self.classes.keys()
            ]
            all_events = []
            for segment_start in range(0, last_event_timestamp, segment_duration):
                segment_end = segment_start + segment_duration
                label = 'NoApnea'
                for timestamp in events_timestamps:
                    start = timestamp[0]
                    end = start + timestamp[1]
                    event_type = timestamp[2]
                    if self._overlaps(segment_start, segment_end, start, end):
                        label = event_type
                        break
                new_entry = (str(label),
                             float(segment_start),
                             float(segment_duration))
                all_events.append(new_entry)
            print(f"Found {len(all_events)} events and non-events for {rml_folder}.")
            self.label_dictionary[rml_folder] = all_events

    def _overlaps(self, segment_start, segment_end, label_start, label_end) -> bool:
        """This function checks if at last 50% of a labelled event is covered by
        :return: True / False"""
        mid = (label_start + label_end) // 2
        return (mid < segment_end < label_end) or (mid > segment_start > label_start)

    def _read_out_single_edf_file(self, edf_folder: str) -> np.array:
        """
        Reads out all files from a single edf directory and concatenates the channel information
        in a numpy array.
        :returns: None
        """
        edf_files: list = sorted([os.path.join(edf_folder, file) for file in os.listdir(edf_folder)])

        full_readout: np.ndarray = np.array([])

        for edf_file in edf_files:
            print(f'Starting readout for file {edf_file}')
            f = pyedflib.EdfReader(edf_file)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sound_data = None
            for i in np.arange(n):
                if signal_labels[i] == self.data_channels:
                    sound_data = f.readSignal(i)
                    break
            f._close()

            if sound_data is None:
                raise ValueError(f"Channel '{self.data_channels}' not found in EDF file")
            full_readout = np.append(full_readout, sound_data)
            print(len(full_readout)/48000)
            del f, sound_data

        return full_readout

    def _get_edf_segments_from_labels(self, edf_folder) -> None:
        """
        Load edf files and create segments by timestamps.
        :return:
        """
        logging.info('5 --- Create segments ---')
        clip_length_seconds = self.clip_length


        print(f"Starting to create segments for user {edf_folder}")
        edf_folder_path = os.path.join(self.edf_preprocess_path, edf_folder)
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

    def _save_segments_as_npz(self, patient_id) -> None:
        """
        Goes through the segments dictionary and creates npz files to save to disk.
        :returns: None
        """
        assert len(self.segments_dictionary) > 0, "No segments available to save."

        data = self.segments_dictionary[patient_id]
        npz_file_name = f"{patient_id}.npz"
        save_path = os.path.join(self.npz_path, npz_file_name)
        arrays = {f"array_{i}": array for i, (_, array) in enumerate(data)}
        labels = np.array([label for label, _ in data])
        np.savez(save_path, labels=labels, **arrays)

    def _load_segments_from_npz(self, npz_file) -> list:
        """
        Load npz files and return labels, values as tuples.
        :returns: list
        """
        load_path = os.path.join(self.npz_path, npz_file)
        loaded = np.load(load_path, allow_pickle=True)
        labels = loaded['labels']
        arrays = [loaded[f"array_{i}"] for i in range(len(labels))]
        data = list(zip(labels, arrays))
        return data

    def _generate_spectrogram(self, wav_path, output_dir) -> None:
        """
        This function takes a single wav file and creates a de-noised spectrogram.
        """
        y, sr = librosa.load(wav_path, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure()
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(output_dir, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _create_all_spectrogram_files(self) -> None:
        """
        Goes through all wav files in the data/processed/audio subfolder and calls the save_spectrogram
        function."""
        logging.info('7 --- Creating spectrogram files ---')

        retired_audio = os.path.join(self.retired_path, 'audio')
        os.makedirs(retired_audio, exist_ok=True)

        for index, wav_file in tqdm(enumerate(os.listdir(self.audio_path))):
            wav_path = os.path.join(self.audio_path, wav_file)
            spec_file_name = wav_file.split('.wav')[0]
            dest_path = os.path.join(self.spectrogram_path, spec_file_name)
            self._generate_spectrogram(wav_path, output_dir=dest_path)
            shutil.move(wav_path, os.path.join(retired_audio, wav_file))

    def _train_val_test_split_spectrogram_files(self) -> None:
        """
        This function shuffles the spectrogram files randomly and assigns them to train, validation or test folders
        according to the ratio defined in train_val_test_ratio.
        :returns: None
        """
        logging.info('8 --- Splitting into train, validation and test ---')
        random.seed(42)
        spectrogram_files = [file for file in os.listdir(self.spectrogram_path) if file.endswith('.png')]
        number_of_files = len(spectrogram_files)
        assert number_of_files > 0, "No files found."

        train, validation, test = [], [], []

        for class_label in self.classes.keys():
            class_files = [file for file in spectrogram_files
                           if file.split('.png')[0].split('_')[2] == class_label]
            number_of_files_in_class = len(class_files)
            random.shuffle(class_files)
            train_index = int(np.floor(number_of_files_in_class * self.train_size))
            validation_index = int(np.floor((number_of_files_in_class - train_index) / 2)) + train_index
            train_files_in_class = class_files[:train_index]
            validation_files_in_class = class_files[train_index:validation_index]
            test_files_in_class = class_files[validation_index:]
            train.extend(train_files_in_class)
            validation.extend(validation_files_in_class)
            test.extend(test_files_in_class)

        random.shuffle(train)
        random.shuffle(validation)
        random.shuffle(test)

        self._move_files(train, 'train')
        self._move_files(validation, 'validation')
        self._move_files(test, 'test')

    def reshuffle_train_val_test(self) -> None:
        """
        In case the earthmover diagram shows distribution imbalance, this function re-shuffles the train, test and
        validation files.
        """
        for folder in os.listdir(self.spectrogram_path):
            for file in os.listdir(os.path.join(self.spectrogram_path, folder)):
                src_path = os.path.join(self.spectrogram_path, folder, file)
                shutil.move(src=src_path, dst=self.spectrogram_path)

        self._train_val_test_split_spectrogram_files()

    def get_train_val_test_distributions(self) -> dict:
        """
        Print the distributions of labels per train, validation and test.
        :returns: dict
        """
        distributions = {}
        all_folders = os.listdir(self.spectrogram_path)

        for folder in all_folders:
            all_files_in_folder = os.listdir(os.path.join(self.spectrogram_path, folder))
            number_of_files = len(all_files_in_folder)
            distributions[folder] = {}
            for file in all_files_in_folder:
                file_label = file.split('_')[2]
                if file_label not in distributions[folder]:
                    distributions[folder][file_label] = 1
                else:
                    distributions[folder][file_label] += 1

            for label in distributions[folder]:
                distributions[folder][label] /= number_of_files
                distributions[folder][label] = round(distributions[folder][label], 2)

        return distributions

    def _move_files(self, files, target_folder) -> None:
        """
        Move files to the target folder.
        :returns: None
        """
        destination_path = os.path.join(self.spectrogram_path, target_folder)
        os.makedirs(destination_path, exist_ok=True)
        number_of_files = len(files)

        for file in files:
            source = os.path.join(self.spectrogram_path, file)
            destination = os.path.join(destination_path, file)
            shutil.move(src=source, dst=destination)

        print(f'Moved {number_of_files} files into /{target_folder}.')

    def _save_to_wav(self, patient_id) -> None:
        """
        This function iterates through the segments list and creates individual wav files.
        :returns: None
        """
        logging.info('6 --- Creating WAV files ---')
        assert len(self.npz_path) > 0, "No npz files to process."

        npz_file = f"{patient_id}.npz"
        data = self._load_segments_from_npz(npz_file=npz_file)
        for index, (label, signal) in enumerate(data):
            file_name = f"{index:05d}_{npz_file.split('.npz')[0]}_{label}.wav"
            write(os.path.join(self.audio_path, file_name), rate=self.sample_rate, data=signal.astype(np.float32))
        shutil.move(os.path.join(self.npz_path, npz_file), os.path.join(self.retired_path, npz_file))

    def get_download_folder_contents(self):
        """
        This function prints all patient IDs for all EDF and RML files currently present in the download folder.
        :returns: None
        """
        edf_folder_contents = os.listdir(self.edf_download_path)
        rml_folder_contents = os.listdir(self.rml_download_path)

        if len(edf_folder_contents) != 0:
            print(f"# --- Found {len(edf_folder_contents)} EDF files with the following ids:")
            edf_ids = set(
                [patient_id.split('-')[0] for patient_id in edf_folder_contents if patient_id.endswith('.edf')]
            )
            print(edf_ids)
        else:
            print("No EDF files found.")

        if len(rml_folder_contents) != 0:
            print(f"# --- Found {len(rml_folder_contents)} RML files with the following ids:")
            rml_ids = set(
                [patient_id.split('-')[0] for patient_id in rml_folder_contents if patient_id.endswith('.rml')]
            )
            print(rml_ids)
        else:
            print("No RML files found.")

    def _collect_processed_raw_files(self, patient_id) -> None:
        """
        This function moves the processed edf and rml files.
        :returns: None
        """
        processed_edf_folders = os.listdir(self.edf_preprocess_path)
        processed_rml_folders = os.listdir(self.rml_preprocess_path)

        for folder in ['edf', 'rml']:
            os.makedirs(os.path.join(self.retired_path, folder), exist_ok=True)


        src_path = os.path.join(self.edf_preprocess_path, patient_id)
        dst_path = os.path.join(self.retired_path, 'edf')
        shutil.move(src=src_path, dst=dst_path)

        src_path = os.path.join(self.rml_preprocess_path, patient_id)
        dst_path = os.path.join(self.retired_path, 'rml')
        shutil.move(src=src_path, dst=dst_path)

    def run(self,
            download: bool = False,
            dictionary: bool = False,
            segments: bool = False,
            create_files: bool = False,
            shuffle: bool = False) -> None:
        """
        Runs the preprocessing pipeline.
        The following parameters can be chosen:
        - download: Set to true to download EDF and RML files provided during the initialization.
        - dictionary: Set to true to create a label dictionary (necessary for following steps).
        - segments: Set to true to create a segments dictionary containing the actual signal (necessary for following steps).
        - create_files: Set to true to create npz, wav and png files.
        - shuffle: Set to true to shuffle files and move into designated train, validation and test folders.
        :return: None
        """
        self._create_directory_structure()

        if download:
            self._download_data()

        self._move_selected_downloads_to_preprocessing()

        patient_ids_to_process = self.ids_to_process if self.ids_to_process is not None \
            else os.listdir(self.rml_preprocess_path)

        for patient_id in patient_ids_to_process:

            if dictionary:
                self._create_sequential_label_dictionary(patient_id)

            if segments:
                self._get_edf_segments_from_labels(patient_id)

            if create_files:
                self._save_segments_as_npz(patient_id)
                self._save_to_wav(patient_id)
                self._collect_processed_raw_files(patient_id)
                self._create_all_spectrogram_files()

        if shuffle:
            self._train_val_test_split_spectrogram_files()

def get_download_urls(file_path, n_ids:int=5, seed=42) -> tuple:
    """
        This function generated the EDF and matching RML URLs needed for the preprocessor.
        How to use: Download the data catalog file for the PSG Audio dataset and put it somewhere in your file
        directory. Specify the catalog file path when calling the function.
        :returns: A tuple with two lists (RML and EDF URLs) as well as a list of unique IDs.
    """
    np.random.seed(seed)
    with open(file_path, 'r') as f:
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

def helper_reset_files(move_from: str = 'retired', move_to: str = 'downloads'):
    for folder_type in ['edf', 'rml']:
        root = os.path.join('data', move_from, folder_type)
        for folder in os.listdir(root):
            for file in os.listdir(os.path.join(root, folder)):
                shutil.move(
                    os.path.join(root, folder, file),
                    os.path.join('data', move_to, folder_type))

    edf_patient_ids = [str(filename.split('-')[0]) for filename in os.listdir(os.path.join('data', 'downloads', 'edf'))]
    rml_patient_ids = [str(filename.split('-')[0]) for filename in os.listdir(os.path.join('data', 'downloads', 'rml'))]
    if set(edf_patient_ids) != set(rml_patient_ids):
        print('Patient IDs do not match')
    print("EDF patient IDs:", set(edf_patient_ids))
    print("RML patient IDs:", set(rml_patient_ids))
