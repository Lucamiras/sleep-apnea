import os
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import librosa
import librosa.feature
from typing import Union, Dict
from src.preprocessing.steps.config import Config
from src.custom_dataclasses.apnea_event import ApneaEvent


class Processor:
    """
    A class responsible for processing extracted signal data and turning them into usable audio features such as
    spectrograms, mel spectrograms and MFCCs.
    """
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

        print("PROCESSOR -- Building audio features for the following IDs:")
        print(patient_ids_to_process)

        # Create spectrograms
        for patient_id in patient_ids_to_process:
            self._load_signals_for_patient_id(patient_id)

    def _load_or_create_dataset(self) -> Dict[str, Union[int, Dict[str, list]]]:
        dataset_present = self.config.dataset_file_name in os.listdir(self.config.signals_path)
        if dataset_present:
            existing_dataset = pickle.load(open(os.path.join(self.config.signals_path,
                                                             self.config.dataset_file_name), "rb"))
            return existing_dataset
        else:
            new_dataset = {
                "clip_length": self.config.clip_length,
                "dataset": {
                    "train": [],
                    "val": [],
                    "test": []
                }
            }
            return new_dataset

    def _load_signals_for_patient_id(self, patient_id:str) -> None:
        pickle_file = f"{patient_id}.pickle"
        processed = []
        data = self._load_segments_from_pickle(pickle_file=pickle_file)
        for index, event in tqdm(enumerate(data['events']), desc=f"Extracting signals for patient {patient_id}"):
            if len(event['signal']) != (self.sample_rate * self.config.clip_length):
                continue
            mel_spectrogram = self._get_audio_features(event['signal'])
            transformed_mel_spectrogram = self._transform_signal_into_image(mel_spectrogram, self.config.image_size)
            apnea_event = ApneaEvent(
                index=index,
                acq_number=patient_id,
                start=event['start'],
                end=event['end'],
                label=event['label'],
                signal=transformed_mel_spectrogram
            )
            if apnea_event.signal.shape == self.config.image_size:
                processed.append(apnea_event)

        self.pickle_processed_data(processed, patient_id)

    def pickle_processed_data(self, processed_data:list, patient_id:str) -> None:
        assert len(processed_data) > 0, "No processed data available to save."
        file_path = f"{os.path.join(self.config.signals_path, patient_id)}.pickle"
        with open(file_path, 'wb') as file:
            pickle.dump(processed_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _transform_signal_into_image(signal_data:np.ndarray, size:tuple) -> np.ndarray:
        img_original = Image.fromarray(signal_data.astype(np.uint8))
        img_resized = transforms.Resize(size)(img_original)
        return np.array(img_resized)

    def _load_segments_from_pickle(self, pickle_file) -> np.ndarray:
        load_path = os.path.join(self.config.pickle_path, pickle_file)
        with open(load_path, 'rb') as file:
            data = pickle.load(file)
            return data

    def _get_audio_features(self, signal_array:list) -> np.ndarray:
        y = np.array(signal_array)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.config.n_mels
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram

    def _check_folder_contains_files(self) -> int:
        return len(os.listdir(self.config.pickle_path)) > 0