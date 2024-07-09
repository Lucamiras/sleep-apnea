import numpy as np
import librosa
import librosa.feature
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
        train_files = os.listdir(self.train_dir)
        train_files = [file.replace('.png', '.wav') for file in train_files]
        train_files = [os.path.join('data', 'retired', 'audio', file) for file in train_files]
        subset_index = int(len(train_files) * self.subset)
        random.shuffle(train_files)
        return train_files[:subset_index]

    def _load_wav_signal(self, file):
        y, sr = librosa.load(file, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=128)
        mel_dB = librosa.power_to_db(mel, ref=np.max)
        return mel_dB

    def _time_mask(self, spec, T=20):
        t = spec.shape[1]
        t0 = random.randint(0, t - T)
        spec[:, t0:t0 + T] = 0
        return spec

    def _freq_mask(self, spec, F=15):
        f = spec.shape[0]
        f0 = random.randint(0, f - F)
        spec[f0:f0 + F, :] = 0
        return spec

    def _aug_pipeline(self, spec, T=20, F=15):
        assert len(self.augmentations) > 0, "No augmentations specified"

        if 'time' in self.augmentations:
            spec = self._time_mask(spec, T)
        if 'frequency' in self.augmentations:
            spec = self._freq_mask(spec, F)

        return spec

    def run(self):

        output_dir = os.path.join('data', 'processed', 'spectrogram', 'augmented')
        os.makedirs(output_dir, exist_ok = True)

        for file in tqdm(self.files_to_augment):
            file_name = 'AUG_' + file.split('/audio/')[1].split('.wav')[0] + '.png'
            mel_dB = self._load_wav_signal(file)
            random_t = np.random.randint(low=20, high=120)
            random_f = np.random.randint(low=10, high=30)
            spec = self._aug_pipeline(mel_dB, T=random_t, F=random_f)
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(spec, sr=self.sample_rate, x_axis='time', y_axis='mel')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight', pad_inches=0)
            plt.close()
