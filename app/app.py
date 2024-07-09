import streamlit as st
import librosa
import librosa.feature
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3, input_size=224):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * int(self.input_size / 2 / 2 / 2) * int(self.input_size / 2 / 2 / 2), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 128 * int(self.input_size / 2 / 2 / 2) * int(self.input_size / 2 / 2 / 2))  # Flatten
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        return x


state_dict_path = 'app/model/model.pth'
# Load your trained model
state_dict = torch.load(state_dict_path)
print(state_dict)
model = CNN(num_classes=3)  # Instantiate your model
model.load_state_dict(state_dict)  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Define the transformation for the spectrogram
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize([0.35885462, 0.11913931, 0.40157405], [0.23603679, 0.10550919, 0.10507757])
])


def preprocess_audio(audio_bytes):
    y, sr = librosa.load(audio_bytes, sr=48_000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(4, 4))
    spectrogram_image = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    transformed = transform(spectrogram_image)  # Add batch dimension
    return transformed

x = preprocess_audio('data/retired/audio/00000_00000995_Hypopnea.wav')
print(x)

# Streamlit app
#st.title("Audio Classification App")
#uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
#
#if uploaded_file is not None:
#    input_tensor = preprocess_audio(uploaded_file)
#
#    with torch.no_grad():
#        output = model(input_tensor)
#        probabilities = torch.softmax(output, dim=1)
#
#    st.write("Classification Probabilities:")
#    st.write(probabilities)
