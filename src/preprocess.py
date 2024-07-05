import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm


class SpecAugmentation:
    def __init__(self, train_dir: str, augmentations: list =['time','frequency'], subset: float = 0.4, sample_rate: int = 48_000) -> None:
        self.train_dir = train_dir
        self.augmentations = augmentations
        self.subset = subset
        self.sample_rate = sample_rate
        self.files_to_augment = self._get_subset_of_files()
        print(f"SpecAugment initialized with {augmentations} and subset size of {subset}.")

    def _get_subset_of_files(self) -> list:
        """
        Returns a subset of files from the train directory.

        This method retrieves a list of files from the train directory, replaces the file extensions,
        appends the file paths, shuffles the list, and returns a subset of the files based on the subset
        percentage specified.

        Returns:
            list: A subset of files from the train directory.

        """
        train_files = os.listdir(self.train_dir)
        train_files = [file.replace('.png', '.wav') for file in train_files]
        train_files = [os.path.join('data', 'retired', 'audio', file) for file in train_files]
        subset_index = int(len(train_files) * self.subset)
        random.shuffle(train_files)
        return train_files[:subset_index]
    
    def _load_wav_signal(self, file) -> np.ndarray:
        """
        Load a WAV file and preprocess the signal.

        Parameters:
            file (str): The path to the WAV file.

        Returns:
            np.ndarray: The preprocessed signal as a numpy array.
        """
        y, _ = librosa.load(file, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=128)
        mel_dB = librosa.power_to_db(mel, ref=np.max)
        return mel_dB

    def _time_mask(self, spec, T=20):
        """
        Applies a time mask to the given spectrogram.

        Parameters:
        spec (numpy.ndarray): The input spectrogram.
        T (int): The length of the time mask. Defaults to 20.

        Returns:
        numpy.ndarray: The spectrogram with the time mask applied.
        """
        t = spec.shape[1]
        t0 = random.randint(0, t - T)
        spec[:, t0:t0 + T] = 0
        return spec

    def _freq_mask(self, spec, F=15):
        """
        Apply frequency masking to the spectrogram.

        Parameters:
        spec (numpy.ndarray): The input spectrogram.
        F (int): The number of frequency bands to mask.

        Returns:
        numpy.ndarray: The spectrogram with frequency masking applied.
        """
        f = spec.shape[0]
        f0 = random.randint(0, f - F)
        spec[f0:f0 + F, :] = 0
        return spec

    def _aug_pipeline(self, spec, T=20, F=15):
        """
        Applies a series of augmentations to the input spectrogram.

        Args:
            spec (numpy.ndarray): The input spectrogram.
            T (int): The time mask parameter. Defaults to 20.
            F (int): The frequency mask parameter. Defaults to 15.

        Returns:
            numpy.ndarray: The augmented spectrogram.
        """
        assert len(self.augmentations) > 0, "No augmentations specified"

        if 'time' in self.augmentations:
            spec = self._time_mask(spec, T)    
        if 'frequency' in self.augmentations:
            spec = self._freq_mask(spec, F)

        return spec
    
    def run(self):
        """
        Runs the preprocessing pipeline to generate augmented spectrogram images.

        This method creates the output directory for the processed spectrogram images,
        iterates over the files to be augmented, loads the WAV signal, applies the
        augmentation pipeline, and saves the resulting spectrogram image.

        Returns:
            None
        """
        output_dir = os.path.join('data', 'processed', 'spectrogram', 'augmented')
        os.makedirs(output_dir, exist_ok=True)

        for file in tqdm(self.files_to_augment):
            file_name = 'AUG_' + file.split('/audio/')[1].replace('.png', '.wav')
            mel_dB = self._load_wav_signal(file)
            spec = self._aug_pipeline(mel_dB)
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(spec, sr=self.sample_rate, x_axis='time', y_axis='mel')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight', pad_inches=0)
            plt.close()
