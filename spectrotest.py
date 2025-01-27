import os
import numpy as np
import librosa.feature
from librosa.util import normalize
import matplotlib.pyplot as plt
from PIL import Image
from mne.io.tag import dtype
from torchvision import transforms
import torch
import cv2
import torchaudio


def load_segments_from_npz(npz_file_path) -> list:
    """
    Load npz files and return labels, values as tuples.
    :returns: list
    """
    load_path = os.path.join(npz_file_path)
    loaded = np.load(load_path, allow_pickle=True)
    labels = loaded['labels']
    arrays = [loaded[f"array_{i}"] for i in range(len(labels))]
    return list(zip(labels, arrays))

path = 'data/preprocess/npz/00000995.npz'

data = load_segments_from_npz(path)
data = [(x, y) for (x, y) in data if x == 'ObstructiveApnea']
label, waveform = data[0]
sample_rate = 48_000
waveform = np.array(waveform)

spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
# MFCC
mfccs = librosa.feature.mfcc(y=waveform, n_mfcc=13, sr=sample_rate)

delta_mfccs = librosa.feature.delta(mfccs)

delta2_mfccs = librosa.feature.delta(mfccs, order=2)

comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
print(mfccs.shape)
print(mel_spectrogram.shape)

transform = transforms.Compose([
    transforms.Resize((224,224))
])

image = Image.fromarray(comprehensive_mfccs.astype(np.uint8))
t_image = transform(image)
plt.imshow(t_image)
plt.show()