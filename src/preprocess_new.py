import os
import shutil
import requests
import mne
import xml.dom.minidom
import numpy as np
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import gc
import random


class Preprocessor:
    """
    This Preprocessor class takes a list of EDF download urls and RML download urls and prepares the files
    for training in the CNN. The following steps are executed:

        1. Creating a data subdirectory in the project folder

        2. Downloading the files and moving them into the subdirectories.

        3. Creating segments from timestamps provided in the rml files.

        4. Creating spectrogram files and randomly shuffling them into train, validation and test folders.
    """
    def __init__(self,
                 project_dir: str,
                 edf_urls: list,
                 rml_urls: list,
                 data_channels: list,
                 classes: dict,
                 sample_rate: int = 48_000):

        self.project_dir = project_dir
        self.edf_urls = edf_urls
        self.rml_urls = rml_urls
        self.data_channels = data_channels
        self.classes = classes
        self.sample_rate = sample_rate
        self.patient_ids = []
        self.label_dictionary = {}
        self.segments_list = []

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        :return: None
        """
        self.edf_path: str = os.path.join(self.project_dir, 'downloads', 'edf')
        self.rml_path: str = os.path.join(self.project_dir, 'downloads', 'rml')
        self.audio_path: str = os.path.join(self.project_dir, 'processed', 'audio')
        self.spectrogram_path: str = os.path.join(self.project_dir, 'processed', 'spectrogram')

        os.makedirs(self.edf_path, exist_ok=True)
        os.makedirs(self.rml_path, exist_ok=True)
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.spectrogram_path, exist_ok=True)

    def _download_data(self) -> None:
        """
        Downloads the files specified in the edf and rml urls.
        :return: None
        """
        print(f"Starting download of {len(self.edf_urls)} EDF files and {len(self.rml_urls)} RML files.")

        for url in tqdm(self.edf_urls):
            response = requests.get(url)
            print(response)
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
        edf_folder_contents = [file for file in os.listdir(self.edf_path) if file.endswith('.edf')]
        rml_folder_contents = [file for file in os.listdir(self.rml_path) if file.endswith('.rml')]
        self.patient_ids = set([uid.split('-')[0] for uid in edf_folder_contents])

        for patient_id in self.patient_ids:
            os.makedirs(os.path.join(self.edf_path, patient_id), exist_ok=True)
            os.makedirs(os.path.join(self.rml_path, patient_id), exist_ok=True)

        for edf_file in edf_folder_contents:
            src_path = os.path.join(self.edf_path, edf_file)
            dst_path = os.path.join(self.edf_path, edf_file.split('-')[0], edf_file)
            shutil.move(src=src_path, dst=dst_path)

        for rml_file in rml_folder_contents:
            src_path = os.path.join(self.rml_path, rml_file)
            dst_path = os.path.join(self.rml_path, rml_file.split('-')[0], rml_file)
            shutil.move(src=src_path, dst=dst_path)

    def _create_label_dictionary(self):
        for rml_folder in os.listdir(self.rml_path):
            for file in os.listdir(os.path.join(self.rml_path, rml_folder)):
                label_path = os.path.join(self.rml_path, rml_folder, file)
                domtree = xml.dom.minidom.parse(label_path)
                group = domtree.documentElement
                events = group.getElementsByTagName("Event")
                events_apnea = []
                non_events_apnea = []
                non_events_counter = 0

                for event in events:
                    event_type = event.getAttribute('Type')
                    if event_type in self.classes.keys():
                        positive_example = (str(event_type),
                                            float(event.getAttribute('Start')),
                                            float(event.getAttribute('Duration')))
                        events_apnea.append(positive_example)

                buffer_in_seconds = 5
                for i in range(len(events_apnea) -1):
                    current_event = events_apnea[i]
                    next_event = events_apnea[i + 1]
                    lower_limit = current_event[1] + current_event[2] + buffer_in_seconds
                    upper_limit = next_event[1]
                    number_of_possible_clips_in_range = (int(upper_limit - lower_limit)) // 10
                    for number_of_clip in range(1, int(number_of_possible_clips_in_range)):
                        negative_example = ('NoApnea',
                                            float(lower_limit + (10 * number_of_clip)),
                                            float(10))
                        non_events_apnea.append(negative_example)
                        non_events_counter += 1
                        if non_events_counter >= len(events_apnea):
                            break
                    if non_events_counter >= len(events_apnea):
                        break

                all_events = events_apnea + non_events_apnea

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
        self.segments_list = []
        clip_length_seconds = 10

        for edf_folder in os.listdir(self.edf_path):
            print(f"Starting to create segments for user {edf_folder}")
            edf_folder_path = os.path.join(self.edf_path, edf_folder)
            edf_readout = self._read_out_single_edf_file(edf_folder_path)

            for label in self.label_dictionary[edf_folder]:
                label_desc = label[0]
                time_stamp = label[1]
                start_idx = int(time_stamp * self.sample_rate)
                end_idx = int((time_stamp + clip_length_seconds) * self.sample_rate)
                segment = edf_readout[start_idx:end_idx]
                if len(segment) > 0:
                    self.segments_list.append((label_desc, segment))

            del edf_readout, segment
            gc.collect()

    def _create_spectrogram_files(self) -> None:
        """
        This function iterates through the segments list and creates spectrogram files from the data.
        :returns: None
        """
        for index, annotated_segment in enumerate(self.segments_list):
            label, signal = annotated_segment
            class_index = self.classes[label]
            file_name = f'spect_{index:05d}_{label}_{class_index}.png'

            denoised_signal = self._spectral_gate(signal, sr=self.sample_rate)
            matrix_d = librosa.stft(denoised_signal)
            db_scaled_spectrogram = librosa.amplitude_to_db(np.abs(matrix_d), ref=np.max)

            plt.figure(figsize=(2, 2))
            librosa.display.specshow(db_scaled_spectrogram, sr=self.sample_rate, x_axis='time', y_axis='log')
            plt.axis('off')
            plt.savefig(os.path.join(self.spectrogram_path, file_name), bbox_inches='tight', pad_inches=0)
            plt.close()

    def _spectral_gate(self, signal, sr, n_fft=2048, hop_length=512, win_length=2048):
        stft_signal = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        magnitude, phase = librosa.magphase(stft_signal)
        noise_profile = np.mean(magnitude[:, 10:], axis=1, keepdims=True)  # Assuming first 10 frames are noise
        threshold = 2 * noise_profile  # This factor can be adjusted
        gated_magnitude = np.maximum(magnitude - threshold, 0)
        denoised_signal = librosa.istft(gated_magnitude * phase, hop_length=hop_length, win_length=win_length)
        
        return denoised_signal

    def _train_val_test_split_spectrogram_files(self) -> None:
        """
        This function shuffles the spectrogram files randomly and assigns them to train, validation or test folders
        according to the ratio defined in train_val_test_ratio.
        """
        train_size = .8
        validation_size = (1 - train_size) / 2
        spectrogram_files = [file for file in os.listdir(self.spectrogram_path) if file.endswith('.png')]
        random.shuffle(spectrogram_files)
        total_number_of_files = len(spectrogram_files)

        train_index = int(total_number_of_files * train_size) # 80
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

        for file in files:
            source = os.path.join(self.spectrogram_path, file)
            destination = os.path.join(destination_path, file)
            shutil.move(src=source, dst=destination)

    def _create_wav_data(self, pcm_rate: int = 32768) -> None:
        """
        This function iterates through the segments list and creates individual wav files.
        :returns: None
        """
        for index, annotated_segment in enumerate(self.segments_list):
            label, signal = annotated_segment
            class_index = self.classes[label]
            file_name_wav = f'{index:05d}_{label}_{class_index}.wav'

            audio_data = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
            audio_data_pcm = np.int16(audio_data * pcm_rate)
            wav_path = os.path.join(self.audio_path, file_name_wav)
            write(wav_path, self.sample_rate, audio_data_pcm)

    def run(self) -> None:
        """os.makedirs(self.parquet_path, exist_ok=True)
        Runs the preprocessing pipeline.
        :return: None
        """
        self._create_directory_structure()
        #self._download_data()
        #self._organize_downloads()
        #self._create_label_dictionary()
        #self._get_edf_segments_from_labels()
        #self._create_wav_data()
        #self._create_spectrogram_files()
        self._train_val_test_split_spectrogram_files()
