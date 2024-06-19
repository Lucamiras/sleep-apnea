import os
import requests
import mne
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import write
import xml.dom.minidom
from pydub import AudioSegment
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt


class Preprocessor:
    def __init__(self, download_dir: str, processed_dir: str, edf_urls: list, rml_urls: list,
                 data_channels: list = None, apnea_types: list = None):
        self.download_dir = download_dir
        self.processed_dir = processed_dir
        self.edf_urls = edf_urls
        self.rml_urls = rml_urls
        self.data_channels = data_channels
        self.apnea_types = apnea_types
        self.edf_path = os.path.join(self.download_dir, 'edfs')
        self.rml_path = os.path.join(self.download_dir, 'rmls')
        self.audio_path = os.path.join(self.processed_dir, 'audio')
        self.spectrogram_path = os.path.join(self.processed_dir, 'spectrogram')
        self.files_dictionary = {}
        self.label_dictionaries = {}
        self.class_labels = {}

    def _make_directories(self) -> None:
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.edf_path, exist_ok=True)
        os.makedirs(self.rml_path, exist_ok=True)
        for subfolder in ['apnea', 'no_apnea']:
            os.makedirs(os.path.join(self.audio_path, subfolder), exist_ok=True)
        for subfolder in ['all', 'apnea', 'no_apnea']:
            os.makedirs(os.path.join(self.spectrogram_path, subfolder), exist_ok=True)

    def _download_data(self) -> None:
        edfs = self.edf_urls
        rmls = self.rml_urls
        for url in edfs:
            response = requests.get(url)
            file_name = url[-28:]
            file_path = os.path.join(self.edf_path, file_name).replace('%5B', '[').replace('%5D', ']')
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        for url in rmls:
            response = requests.get(url)
            file_name = url[-19:]
            file_path = os.path.join(self.rml_path, file_name).replace('%5B', '[').replace('%5D', ']')
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print("Download of labels completed successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

    def _read_edf_data(self):
        edfs = os.listdir(self.edf_path)
        edf_user_ids = set([uid.split('[')[0] for uid in edfs if uid.endswith('.edf')])
        for user_id in edf_user_ids:
            files = sorted([file for file in edfs if file.startswith(user_id)])
            self.files_dictionary[user_id] = files

    def _create_parquet_files(self) -> np.array:
        for user_id, files in self.files_dictionary.items():
            patient_data = np.array([])
            for file in files:
                raw = mne.io.read_raw(os.path.join(self.edf_path, file), verbose=0, preload=False)
                length = int(len(raw) / 10)
                raw_selected = raw.pick(self.data_channels)
                del raw
                gc.collect()
                for i in tqdm(range(10)):
                    start = length * i
                    end = length * (i + 1)
                    new_data = raw_selected.get_data(start=start, stop=end)
                    patient_data = np.append(patient_data, new_data)
                    del new_data
                    gc.collect()
            df = pd.DataFrame(patient_data.T, columns=self.data_channels)
            df.to_parquet(self.processed_dir + f'/{user_id}.parquet.gzip', index=False, compression='gzip')
            del raw_selected, patient_data, df
            gc.collect()

    def _create_wav_file(self, sample_rate, pcm_rate, data_slice: int = None) -> None:
        parquet_files = [x for x in os.listdir(self.processed_dir) if x.endswith('gzip')]
        column_names = {str(i): str(channel_name) for i, channel_name in enumerate(self.data_channels)}

        for p_file in parquet_files:
            in_path = os.path.join(self.processed_dir, p_file)
            mic_data = pd.read_parquet(in_path)
            mic_data.rename(columns=column_names, inplace=True)

            mic_data_sample = mic_data[:data_slice]

            del mic_data
            gc.collect()

            audio_data = mic_data_sample.to_numpy()
            audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), (-1, 1))
            audio_data_pcm = np.int16(audio_data * pcm_rate)

            sample_rate = sample_rate
            wav_path = os.path.join(self.processed_dir, p_file.split('.')[0] + '.wav')
            write(wav_path, sample_rate, audio_data_pcm)

    def _create_label_dictionary(self):
        for rml_file in os.listdir(self.rml_path):
            label_path = os.path.join(self.rml_path, rml_file)
            domtree = xml.dom.minidom.parse(label_path)
            group = domtree.documentElement
            events = group.getElementsByTagName("Event")
            events_apnea = []

            for event in events:
                event_type = event.getAttribute('Type')
                if event_type in self.apnea_types:
                    iter_type_start_duration = (str(event_type),
                                                float(event.getAttribute('Start')),
                                                float((event.getAttribute('Duration'))))
                    events_apnea.append(iter_type_start_duration)

            self.label_dictionaries[rml_file] = events_apnea

    def _create_anti_labels(self):
        neg_labels = []
        for rml_file in os.listdir(self.rml_path):
            labels = self.label_dictionaries[rml_file]
            counter = 0
            for i in range(len(labels) - 1):
                lower_limit = labels[i][1] + 15
                upper_limit = labels[i + 1][1]
                number_of_clips = (int(upper_limit - lower_limit)) // 10
                for j in range(1, int(number_of_clips)):
                    neg_label = ('No apnea', (lower_limit + (10 * j)))
                    counter += 1
                    neg_labels.append(neg_label)
                    if counter >= len(labels):
                        break
                if counter >= len(labels):
                    break
            self.label_dictionaries[rml_file] += neg_labels

    def _create_clips(self) -> None:
        wav_files = [x for x in os.listdir(self.processed_dir) if x.endswith('.wav')]

        for wav_file in wav_files:
            wav_file_name = wav_file.split('.wav')[0]
            wav_file_path = os.path.join(self.processed_dir, wav_file)
            audio = AudioSegment.from_file(wav_file_path)
            clip_length_ms = 10 * 1000

            for i, label in enumerate(self.label_dictionaries[wav_file_name + '.rml']):
                file_name = str(f"{i}_{label[0]}")
                start_ms = int(label[1] / 60) * 1000
                end_ms = int(start_ms + clip_length_ms)
                if end_ms < (audio.duration_seconds * 1000):
                    split_audio = audio[start_ms:end_ms]
                    if label[0] == 'No apnea':
                        out_path = os.path.join(self.audio_path, 'no_apnea', file_name + '.wav')
                    else:
                        out_path = os.path.join(self.audio_path, 'apnea', file_name + '.wav')
                    split_audio.export(out_path, format="wav")

    def _save_spectrogram(self, audio_path, save_path) -> None:
        y, sr = librosa.load(audio_path, sr=None)
        y = y / np.max(np.abs(y))
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_amplified = y_denoised * 10
        d = librosa.stft(y_amplified)
        d_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)

        plt.figure(figsize=(2, 2))
        librosa.display.specshow(d_db, sr=sr, x_axis='time', y_axis='log')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _create_spectrograms(self):
        class_dirs = os.listdir(self.audio_path)
        for i, class_dir in enumerate(class_dirs):
            self.class_labels[i] = class_dir
            class_path = os.path.join(self.audio_path, class_dir)
            for j, audio_file in tqdm(enumerate(os.listdir(class_path))):
                audio_file_path = os.path.join(class_path, audio_file)
                save_path = os.path.join(
                    os.path.join(self.spectrogram_path, 'all', f'spectrogram_{j:0>4}_{class_dir}_{i}.png'))
                self._save_spectrogram(audio_file_path, save_path)

    def preprocess_source_data(self, download: bool = False, preprocess: bool = False, sr: int = 48000,
                               pcm_rate: int = 32768):
        self._make_directories()
        if download:
            self._download_data()
        if not download:
            if len(os.listdir(self.edf_path)) == 0:
                raise Exception("When selecting download = False, please make sure to have files in your download_dir.")
        if preprocess:
            self._read_edf_data()
            self._create_parquet_files()
            self._create_wav_file(sr, pcm_rate)
        self._create_label_dictionary()
        self._create_anti_labels()
        self._create_clips()
        self._create_spectrograms()
