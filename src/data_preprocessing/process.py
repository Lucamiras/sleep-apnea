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
from src.data_preprocessing.config import Config


class Processor:
    """
    A class responsible for processing extracted signal data and turning them into usable audio features such as
    spectrograms, mel spectrograms and MFCCs.
    """
    signal_list = []
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
            else [npz_filename.split('.npz')[0] for npz_filename in os.listdir(self.config.pickle_path)]

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
        self._build_dataset()

    def _load_or_create_dataset(self):
        dataset_present = self.config.dataset_file_name in os.listdir(self.config.signals_path)
        if dataset_present:
            existing_dataset = pickle.load(open(os.path.join(self.config.signals_path,
                                                             self.config.dataset_file_name), "rb"))
            return existing_dataset
        if not dataset_present:
            new_dataset = {
                "clip_length": self.config.clip_length,
                "dataset": {
                    "train": [],
                    "val": [],
                    "test": []
                }
            }
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
        pickle_file = f"{patient_id}.pickle"
        data = self._load_segments_from_pickle(pickle_file=pickle_file)
        for index, event in tqdm(enumerate(data['events']), desc=f"Extracting signals for patient {patient_id}"):
            apnea_event = {
                'index': index,
                'acq_number': patient_id,
                'start': event['start'],
                'end': event['end'],
                'label': event['label'],
                'signal': event['signal']
            }
            if len(apnea_event['signal']) == self.config.clip_length * self.sample_rate:
                self.signal_list.append(apnea_event)

    def _build_dataset(self) -> None:
        for signal_list, split in zip(
                [self.train_signals, self.val_signals, self.test_signals],
                ['Train', 'Val', 'Test']):
            for sample in tqdm(signal_list, desc=f"Generating features for {split} signals ..."):
                mel_spectrogram = self._get_audio_features(sample['signal'])
                transformed_mel_spectrogram = self._transform_signal_into_image(mel_spectrogram, self.config.image_size)
                sample['signal'] = transformed_mel_spectrogram
                self.spectrogram_dataset.get('dataset').get(split.lower()).append(sample)

    @staticmethod
    def _transform_signal_into_image(signal_data:np.ndarray, size:tuple):
        img_original = Image.fromarray(signal_data.astype(np.uint8))
        img_resized = transforms.Resize(size)(img_original)
        return np.array(img_resized)

    def _load_segments_from_pickle(self, pickle_file):
        load_path = os.path.join(self.config.pickle_path, pickle_file)
        with open(load_path, 'rb') as file:
            data = pickle.load(file)
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
        return len(os.listdir(self.config.pickle_path)) > 0

    def _shuffle_and_split(self) -> None:
        # calculate train val test split indices
        number_of_samples = len(self.signal_list)
        assert number_of_samples > 0, "No samples found."
        train_index = int(np.round(number_of_samples * self.config.train_size))
        val_index = int(np.round(number_of_samples) * (1 - self.config.train_size) / 2) + train_index

        # load and shuffle signals
        random.shuffle(self.signal_list)

        # create datasets
        self.train_signals = self.signal_list[:train_index]
        self.val_signals = self.signal_list[train_index:val_index]
        self.test_signals = self.signal_list[val_index:]

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