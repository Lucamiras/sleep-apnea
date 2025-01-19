import os
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import librosa
import librosa.feature
import random
from scipy.io import wavfile
from scipy.signal import resample
from src.dataclasses.spectrogram import Spectrogram, SpectrogramDataset
from src.data_preprocessing.config import Config


class Processor:
    """
    A class responsible for processing extracted signal data and turning them into usable audio features such as
    spectrograms, mel spectrograms and MFCCs.
    """
    signal_dictionary = {}
    augment_chunks = []
    train_signals = None
    val_signals = None
    test_signals = None

    def __init__(self, config:Config):
        self.config = config
        self.sample_rate = self.config.new_sample_rate if self.config.new_sample_rate else self.config.sample_rate
        self.spectrogram_dataset = self._load_or_create_dataset()

    def process(self):
        print("PROCESSOR -- Starting processing")

        # Check if npz files exist
        if not self._check_folder_contains_files():
            raise Exception("No NPZ files to process.")

        # Select patient IDs to process according to Config.ids_to_process
        patient_ids_to_process = self.config.ids_to_process if self.config.ids_to_process is not None \
            else [npz_filename.split('.npz')[0] for npz_filename in os.listdir(self.config.npz_path)]

         # Check if augmentation should apply
        augment = True if isinstance(self.config.augment_ratio, float) else False
        if augment:
            self._load_and_chunk_ambient_wav_files()

        print("PROCESSOR -- Building audio features for the following IDs:")
        print(patient_ids_to_process)

        # Create spectrograms
        for patient_id in patient_ids_to_process:
            self._load_signals_for_patient_id(patient_id)

        # Shuffle and split into train, val and test
        self._shuffle_and_split()

        # Create spectrograms
        self._transform_signals_into_features(augment=augment)

    def _load_or_create_dataset(self):
        dataset_present = self.config.dataset_file_name in os.listdir(self.config.signals_path)
        if dataset_present:
            existing_dataset = pickle.load(open(os.path.join(self.config.signals_path,
                                                             self.config.dataset_file_name), "rb"))
            return existing_dataset
        if not dataset_present:
            new_dataset = SpectrogramDataset(
                clip_length=self.config.clip_length, data={"train":[], "val":[], "test":[]})
            return new_dataset

    def _augment(self, data, label):
        overlay_data_index = int(np.random.choice(len(self.augment_chunks), 1))
        overlay_data = self.augment_chunks[overlay_data_index]
        overlay_amplitude = np.random.choice(range(1, 8)) / 10
        augmented_data = self.overlay_sound(data, overlay_data, overlay_amplitude)
        label_split = label.split('_')
        label_split.insert(1, 'AUG')
        augmented_label = '_'.join(label_split)
        return augmented_data, augmented_label

    def _load_signals_for_patient_id(self, patient_id:str):
        npz_file = f"{patient_id}.npz"
        data = self._load_segments_from_npz(npz_file=npz_file)
        for index, arr in enumerate(tqdm(data, desc=f"Extracting signals for patient {patient_id}")):
            spec_file_name = f"{index:05d}_{patient_id}_{arr[0]}"
            if arr[1].shape[0] == self.config.clip_length * self.sample_rate:
                self.signal_dictionary[spec_file_name] = arr[1]

    def _transform_signals_into_features(self, augment:bool=False):
        for signal_dict, desc in zip([self.train_signals, self.val_signals, self.test_signals], ['Train', 'Val', 'Test']):
            for signal in tqdm(list(signal_dict.items()), desc=f"Generating features for {desc} signals ..."):
                label, data = signal
                signal_dict[label] = self._get_audio_features(data)
                if not augment:
                    continue
                if signal_dict is not self.train_signals:
                    continue
                if np.random.choice(10) / 10 < self.config.augment_ratio:
                    augmented_data, augmented_label = self._augment(data, label)
                    signal_dict[augmented_label] = self._get_audio_features(augmented_data)

    def _build_dataset(self) -> None:
        for signal_dictionary, split in zip(
                [self.train_signals, self.val_signals, self.test_signals],
                ['Train', 'Val', 'Test']):
            for sample in tqdm(list(signal_dictionary.items()), desc=f"Generating features for {split} signals ..."):
                label, signal_data = sample
                _, patient_id, classification = label.split('_')
                signal_data = self._transform_signal_into_image(signal_data, self.config.image_size)
                new_spectrogram = Spectrogram(
                    label=classification,
                    patient_id=patient_id,
                    data=signal_data)
                self.spectrogram_dataset.data.get(split.lower()).append(new_spectrogram)
        return None

    @staticmethod
    def _transform_signal_into_image(signal_data:np.ndarray, size:tuple):
        img_original = Image.fromarray(signal_data.astype(np.uint8))
        img_resized = transforms.Resize(size)(img_original)
        return np.array(img_resized)

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
            sr=self.sample_rate,
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

    def _load_and_chunk_ambient_wav_files(self):
        ambient_files = os.listdir(self.config.ambient_noise_path)
        clip_length = (self.config.new_sample_rate
                       if self.config.new_sample_rate
                       else self.config.sample_rate) * self.config.clip_length
        step_size = (self.config.new_sample_rate
                       if self.config.new_sample_rate
                       else self.config.sample_rate) * 20

        for ambient_file in tqdm(ambient_files, desc="PROCESSOR -- Creating ambient chunks for data augmentation"):
            file_path = os.path.join(self.config.ambient_noise_path, ambient_file)
            sample_rate, audio_data = wavfile.read(file_path)
            if sample_rate != self.config.sample_rate:
                audio_data = resample(audio_data, clip_length)
            audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float16) / np.max(np.abs(audio_data))
            audio_data = audio_data.reshape(-1,)
            sliced_clips = [audio_data[i:i+clip_length] for i in range(0,len(audio_data),step_size)
                            if len(audio_data[i:i+clip_length]) == clip_length]
            self.augment_chunks.extend(sliced_clips)

    @staticmethod
    def overlay_sound(original_data, overlay_data, overlay_amplitude=0.5):
        augmented_data = original_data + (overlay_amplitude * overlay_data)
        augmented_data = np.clip(augmented_data, -1.0, 1.0)
        return augmented_data