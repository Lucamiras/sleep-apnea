import streamlit as st
import librosa
import librosa.feature
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


# -- | App layout | -- #
st.title('Deep Sleep Audio Analysis')
uploaded_file = st.file_uploader('Upload WAV')


# -- | Functions | -- #
def load_audio(file):
    audio, sample_rate = librosa.load(file, sr=None)
    return audio, sample_rate


def split_audio(audio, sample_rate, chunk_duration):
    chunk_length = chunk_duration * sample_rate
    return [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]


def generate_spectrogram(y, sample_rate, n_mels=20):
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def spectrogram_to_tensor(spectrogram):
    spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
    tensor = torch.tensor(spectrogram, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    return tensor


def plot_spectrogram(spectrogram, sr):
    plt.figure(figsize=(7, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

# -- | App logic | -- #
sliding_window = 5
step_size = 2
# first iteration: start 0 + sliding window
# second iteration: start 2 + sliding window
# third iteration: start 4
if uploaded_file:
    tic = time.time()
    audio, sample_rate = load_audio(uploaded_file)
    mel_db = generate_spectrogram(audio, sample_rate=sample_rate)
    tensor = spectrogram_to_tensor(mel_db)
    spectrogram = plot_spectrogram(mel_db, sample_rate)
    st.pyplot(spectrogram)
    toc = time.time()
    st.write(tensor)
    st.write(toc - tic)



