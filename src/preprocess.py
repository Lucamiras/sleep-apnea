import os
import shutil
import h5py
import requests
import xml.dom.minidom
import numpy as np
from tqdm import tqdm
import librosa
import librosa.feature
import gc
import random
import logging
import pyedflib
import re
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class Config:
    """
    Configuration class for managing all settings and paths required by the data processing pipeline.

    Attributes:
        project_dir (str): Root directory where all preprocessing will take place.
        classes (dict): Dictionary with class names and integer values. Example: {"NoApnea":0, "ObstructiveApnea":1}.
        download_files (bool): Specify if the preprocessor should include the downloading step.
        extract_signals (bool): Specify if the preprocessor should include the extraction step.
        process_signals (bool): Specify if the preprocessor should include the processing step.
        serialize_signals (bool): Specify if the preprocessor should include the serialization step.
        overrides (dict): Overrides of any other parameters can be changed via the overrides method by passing a
        dictionary with attribute names as keys and parameters as values.
        edf_urls (list): A list with EDF download urls. Default: None
        rml_urls (list): A list with RML download urls. Default: None
        data_channels (str): The data channel to extract from the EDF file. Default: 'Mic'
        edf_step_size (int): Amount of signals to process during EDF extraction. This is necessary as to not overfill
        memory. Default: 10_000_000
        sample_rate (int): The appropriate sample rate for the signal. Default: 48_000
        clip_length (int): Length of extracted samples that will be used for training. Default: 30
        ids_to_process (list): List of IDs in case not all files from download / preprocess folders should be considered.
        Default: None
        train_size (float): How much of the total samples should be considered for training. Default: 0.8
        edf_download_path (str): Path in root folder.
        rml_download_path (str): Path in root folder.
        edf_preprocess_path (str): Path in root folder.
        rml_preprocess_path (str): Path in root folder.
        npz_path (str): Path in root folder.
        audio_path (str): Path in root folder.
        spectrogram_path (str): Path in root folder.
        signals_path (str): Path in root folder.
        retired_path (str): Path in root folder.

    Methods:
        _create_directory_structure():
            Creates the folders in the root directory specified in the project_dir attribute.
    """

    def __init__(self,
                 classes: dict,
                 project_dir: str = 'data',
                 download_files: bool = False,
                 extract_signals: bool = True,
                 process_signals: bool = True,
                 serialize_signals: bool = True,
                 overrides: dict = None):

        if overrides:
            for key, value in overrides.items():
                setattr(self, key, value)

        # Basic inputs
        self.project_dir = project_dir
        self.classes = classes

        # Download config
        self.download_files = download_files
        self.catalog_filepath = os.path.join(self.project_dir, 'file_catalog.txt')
        self.edf_urls = None
        self.rml_urls = None

        # Extract signals
        self.extract_signals = extract_signals
        self.data_channels = 'Mic'
        self.edf_step_size = 10_000_000
        self.sample_rate = 48_000
        self.clip_length = 30
        self.n_mels = 128

        # Process signals
        self.process_signals = process_signals
        self.ids_to_process = None

        # Serialize signals
        self.serialize_signals = serialize_signals
        self.train_size = 0.8

        # Paths
        self.edf_download_path = os.path.join(self.project_dir, 'downloads', 'edf')
        self.rml_download_path = os.path.join(self.project_dir, 'downloads', 'rml')
        self.ambient_noise_path = os.path.join(self.project_dir, 'downloads', 'ambient')
        self.edf_preprocess_path = os.path.join(self.project_dir, 'preprocess', 'edf')
        self.rml_preprocess_path = os.path.join(self.project_dir, 'preprocess', 'rml')
        self.npz_path = os.path.join(self.project_dir, 'preprocess', 'npz')
        self.audio_path = os.path.join(self.project_dir, 'processed', 'audio')
        self.spectrogram_path = os.path.join(self.project_dir, 'processed', 'spectrogram')
        self.signals_path = os.path.join(self.project_dir, 'processed', 'signals')
        self.retired_path = os.path.join(self.project_dir, 'retired')
        self._create_directory_structure()

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        returns: None
        """
        os.makedirs(self.edf_download_path, exist_ok=True)
        os.makedirs(self.rml_download_path, exist_ok=True)
        os.makedirs(self.edf_preprocess_path, exist_ok=True)
        os.makedirs(self.rml_preprocess_path, exist_ok=True)
        os.makedirs(self.npz_path, exist_ok=True)
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.spectrogram_path, exist_ok=True)
        os.makedirs(self.signals_path, exist_ok=True)
        os.makedirs(self.retired_path, exist_ok=True)

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

class Extractor:
    """
    A class responsible for extracting signal data from EDF files using labels from RML files .

    Attributes:
        config (Config): An instance of the Config class containing download settings and file paths.

    Methods:
        extract():
            Starts the extraction process and returns a segment dictionary saved as npz.

    """
    def __init__(self, config):
        self.config = config
        self.label_dictionary = {}
        self.segments_dictionary = {}

    def extract(self):
        print("EXTRACTOR -- Starting extraction ...")
        # check if files are in download folder
        if self._check_folder_contains_files(self.config.edf_download_path, self.config.rml_download_path):
            self._move_selected_downloads_to_preprocessing()

        # check if preprocess folder is empty
        if not self._check_folder_contains_files(self.config.edf_preprocess_path, self.config.rml_preprocess_path):
            raise Exception("No EDF or RML files found in preprocess folder.")

        # for each patient_id in preprocessing:
        if not self._match_ids(self.config.edf_preprocess_path, self.config.rml_preprocess_path):
            raise Exception("Not every EDF files has an RML file match.")

        print("EXTRACTOR -- All tests passed.")

        patient_ids_to_process = self.config.ids_to_process if self.config.ids_to_process is not None \
            else os.listdir(self.config.rml_preprocess_path)

        print("EXTRACTOR -- Starting extraction for the following IDs:")
        print(patient_ids_to_process)

        for patient_id in patient_ids_to_process:
            ## get label dictionary
            self._create_sequential_label_dictionary(patient_id)
            ## create segments
            self._get_edf_segments_from_labels(patient_id)
            ## saves as npz
            self._save_segments_as_npz(patient_id)

    def _move_selected_downloads_to_preprocessing(self):
        """
        Moves files into folders by patient id.
        :returns:
        """
        edf_folder_contents = [file for file in os.listdir(self.config.edf_download_path) if file.endswith('.edf')]
        rml_folder_contents = [file for file in os.listdir(self.config.rml_download_path) if file.endswith('.rml')]

        if self.config.ids_to_process is not None:
            edf_folder_contents = [file for file in edf_folder_contents if file.split('-')[0] in self.config.ids_to_process]
            rml_folder_contents = [file for file in rml_folder_contents if file.split('-')[0] in self.config.ids_to_process]

        unique_edf_file_ids = set([file.split('-')[0] for file in edf_folder_contents])
        unique_rml_file_ids = set([file.split('-')[0] for file in rml_folder_contents])

        assert unique_edf_file_ids == unique_rml_file_ids, ("Some EDF or RML files don't have matching pairs. "
                                                            "Preprocessing will not be possible:"
                                                            f"{unique_edf_file_ids}, {unique_rml_file_ids}")

        print(f"Preprocessing the following IDs: {unique_edf_file_ids}")

        for unique_edf_file_id in unique_edf_file_ids:
            os.makedirs(os.path.join(self.config.edf_preprocess_path, unique_edf_file_id), exist_ok=True)

        for unique_rml_file_id in unique_rml_file_ids:
            os.makedirs(os.path.join(self.config.rml_preprocess_path, unique_rml_file_id), exist_ok=True)

        for edf_file in edf_folder_contents:
            src_path = os.path.join(self.config.edf_download_path, edf_file)
            dst_path = os.path.join(self.config.edf_preprocess_path, edf_file.split('-')[0], edf_file)
            shutil.move(src=src_path, dst=dst_path)

        for rml_file in rml_folder_contents:
            src_path = os.path.join(self.config.rml_download_path, rml_file)
            dst_path = os.path.join(self.config.rml_preprocess_path, rml_file.split('-')[0], rml_file)
            shutil.move(src=src_path, dst=dst_path)

    def _get_edf_segments_from_labels(self, edf_folder) -> None:
        """
        Load edf files and create segments by timestamps.
        :return:
        """
        clip_length_seconds = self.config.clip_length

        print(f"Starting to create segments for patient {edf_folder}")
        edf_folder_path = os.path.join(self.config.edf_preprocess_path, edf_folder)
        edf_readout = self._read_out_single_edf_file(edf_folder_path)
        self.segments_dictionary[edf_folder] = []

        for label in self.label_dictionary[edf_folder]:
            label_desc = label[0]
            time_stamp = label[1]
            start_idx = int(time_stamp * self.config.sample_rate)
            end_idx = int((time_stamp + clip_length_seconds) * self.config.sample_rate)
            segment = edf_readout[start_idx:end_idx]
            if len(segment) > 0:
                if self._no_change(segment):
                    break
                else:
                    self.segments_dictionary[edf_folder].append((label_desc, segment))

        del edf_readout, segment
        gc.collect()

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
                if signal_labels[i] == self.config.data_channels:
                    sound_data = f.readSignal(i)
                    break
            f._close()

            if sound_data is None:
                raise ValueError(f"Channel '{self.config.data_channels}' not found in EDF file")
            full_readout = np.append(full_readout, sound_data)
            print(len(full_readout)/self.config.sample_rate)
            del f, sound_data

        return full_readout

    def _create_sequential_label_dictionary(self, rml_folder:str) -> None:
        """
        This function goes through the EDF file in chunks of a determined size, i.e. 30 seconds,
        and labels each chunk according to the provided RML file.
        """
        file = os.listdir(os.path.join(self.config.rml_preprocess_path, rml_folder))[0]
        label_path = os.path.join(self.config.rml_preprocess_path, rml_folder, file)
        domtree = xml.dom.minidom.parse(label_path)
        group = domtree.documentElement
        events = group.getElementsByTagName('Event')
        last_event_timestamp = int(float(events[-1].getAttribute('Start')))
        segment_duration = self.config.clip_length
        events_timestamps = [
            (float(event.getAttribute('Start')),
             float(event.getAttribute('Duration')),
             event.getAttribute('Type')) for event in events
            if event.getAttribute('Type') in self.config.classes.keys()
        ]
        all_events = []
        for segment_start in range(0, last_event_timestamp, segment_duration):
            segment_end = segment_start + segment_duration
            label = 'NoApnea'
            for timestamp in events_timestamps:
                start = timestamp[0]
                end = start + timestamp[1]
                event_type = timestamp[2]
                if start > segment_end:
                    break
                if self._overlaps(segment_start, segment_end, start, end):
                    label = event_type
                    break
            new_entry = (str(label),
                         float(segment_start),
                         float(segment_duration))
            all_events.append(new_entry)
        print(f"Found {len(all_events)} events and non-events for {rml_folder}.")
        self.label_dictionary[rml_folder] = all_events

    def _save_segments_as_npz(self, patient_id) -> None:
        """
        Goes through the segments dictionary and creates npz files to save to disk.
        :returns: None
        """
        assert len(self.segments_dictionary) > 0, "No segments available to save."

        data = self.segments_dictionary[patient_id]
        npz_file_name = f"{patient_id}.npz"
        save_path = os.path.join(self.config.npz_path, npz_file_name)
        arrays = {f"array_{i}": array for i, (_, array) in enumerate(data)}
        labels = np.array([label for label, _ in data])
        np.savez(save_path, labels=labels, **arrays)

    @staticmethod
    def _overlaps(segment_start, segment_end, label_start, label_end) -> bool:
        """This function checks if at last 50% of a labelled event is covered by
        :return: True / False"""
        mid = (label_start + label_end) / 2
        return segment_start < mid < segment_end

    @staticmethod
    def _no_change(array):
        return (array.sum() / len(array)) == array[0]

    @staticmethod
    def _check_folder_contains_files(edf_folder, rml_folder):
        edf_files_in_folder = os.listdir(edf_folder)
        rml_files_in_folder = os.listdir(rml_folder)
        not_empty = (len(edf_files_in_folder) > 0) and (len(rml_files_in_folder) > 0)
        return not_empty

    @staticmethod
    def _match_ids(edf_folder, rml_folder):
        edf_ids = sorted(os.listdir(edf_folder))
        rml_ids = sorted(os.listdir(rml_folder))
        return edf_ids == rml_ids

class Processor:
    """
    A class responsible for processing extracted signal data and turning them into usable audio features such as
    spectrograms, mel spectrograms and MFCCs.
    """
    train_signals = None
    val_signals = None
    test_signals = None

    def __init__(self, config):
        self.config = config
        self.signal_dictionary = {}

    def process(self):
        print("PROCESSOR -- Starting processing")
        # Check if npz files exist
        if not self._check_folder_contains_files():
            raise Exception("No NPZ files to process.")

        patient_ids_to_process = self.config.ids_to_process if self.config.ids_to_process is not None \
            else [npz_filename.split('.npz')[0] for npz_filename in os.listdir(self.config.npz_path)]

        print("PROCESSOR -- Building audio features for the following IDs:")
        print(patient_ids_to_process)
        for patient_id in patient_ids_to_process:
            self._create_all_spectrogram_files_as_arrays(patient_id)
        self._shuffle_and_split()

    def _create_all_spectrogram_files_as_arrays(self, patient_id:str):
        npz_file = f"{patient_id}.npz"
        data = self._load_segments_from_npz(npz_file=npz_file)
        for index, arr in tqdm(enumerate(data)):
            spec_file_name = f"{index:05d}_{patient_id}_{arr[0]}"
            self.signal_dictionary[spec_file_name] = self._get_audio_features(arr[1])

    def _load_segments_from_npz(self, npz_file) -> list:
        """
        Load npz files and return labels, values as tuples.
        :returns: list
        """
        load_path = os.path.join(self.config.npz_path, npz_file)
        loaded = np.load(load_path, allow_pickle=True)
        labels = loaded['labels']
        arrays = [loaded[f"array_{i}"] for i in range(len(labels))]
        data = list(zip(labels, arrays))
        return data

    def _get_audio_features(self, signal_array:list):
        y = np.array(signal_array)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram

    def _check_folder_contains_files(self):
        return len(os.listdir(self.config.npz_path)) > 0

    def _shuffle_and_split(self) -> None:
        # calculate train val test split indices
        number_of_samples = len(self.signal_dictionary)
        assert number_of_samples > 0, "No samples found."
        train_index = int(np.round(number_of_samples * self.config.train_size))
        val_index = int(np.round(number_of_samples) * (1 - self.config.train_size) / 2) + train_index

        # load and shuffle signals
        signal_data = list(self.signal_dictionary.items())
        random.shuffle(signal_data)

        # create datasets
        train_signals = signal_data[:train_index]
        val_signals = signal_data[train_index:val_index]
        test_signals = signal_data[val_index:]

        # return dictionaries
        self.train_signals = {key: value for key, value in train_signals}
        self.val_signals = {key: value for key, value in val_signals}
        self.test_signals = {key:value for key, value in test_signals}

class Serializer:
    """
    A class responsible for shuffling, splitting and serializing data.
    """
    train_signals = None
    val_signals = None
    test_signals = None

    def __init__(self, config):
        self.config = config

    def serialize(self, train_signals, val_signals, test_signals):
        self.train_signals = train_signals
        self.val_signals = val_signals
        self.test_signals = test_signals
        self._save_dataset_as_hdf5()

    @staticmethod
    def _save_dict_as_hdf5(group, dictionary):
        for key, value in dictionary.items():
            group.create_dataset(
                key,
                data=value,
                compression='gzip',
                chunks=True
            )

    def _save_dataset_as_hdf5(self):
        save_path = os.path.join(self.config.signals_path, 'dataset.h5')
        with h5py.File(save_path, 'w') as hdf:
            train_group = hdf.create_group('Train')
            val_group = hdf.create_group('Val')
            test_group = hdf.create_group('Test')

            self._save_dict_as_hdf5(train_group, self.train_signals)
            self._save_dict_as_hdf5(val_group, self.val_signals)
            self._save_dict_as_hdf5(test_group, self.test_signals)

class DataPreprocessor:
    def __init__(self, downloader, extractor, processor, serializer, config):
        self.downloader = downloader
        self.extractor = extractor
        self.processor = processor
        self.serializer = serializer
        self.config = config

    def run(self):
        if self.config.download_files:
            # This will download any files specified in the EDF_URLS and RML_URLS from the Config class
            self.downloader.download_data()

        if self.config.extract_signals:
            # This will extract audio signals from the EDF files according to the RML files and save them as NPZ files.
            self.extractor.extract()

        if self.config.process_signals:
            # This will process the extracted signals, and process it into audio features
            self.processor.process()

        if self.config.serialize_signals:
            # This will take the processor signals and turn them into shuffled pickle files
            self.serializer.serialize(self.processor.train_signals,
                                      self.processor.val_signals,
                                      self.processor.test_signals)

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
