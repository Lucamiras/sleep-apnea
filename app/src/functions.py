import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import numpy as np
import streamlit as st
from src.dataset import SpectrogramDataset
from src.globals import TARGET_SR, CHUNK_DURATION, N_MELS
import matplotlib.pyplot as plt
import io
from PIL import Image


def load_audio(file, target_sr=TARGET_SR):
    audio, sr = librosa.load(file, sr=target_sr)
    return audio, sr

def split_audio(audio, sr, chunk_duration=CHUNK_DURATION):
    chunk_length = chunk_duration * sr
    return [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

def generate_spectrogram(signal, output_dir) -> None:
    mel = librosa.feature.melspectrogram(y=signal, sr=TARGET_SR, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(mel_db)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_db, sr=TARGET_SR, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(output_dir, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_dataloader_from_chunks(specs, mean, std, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = SpectrogramDataset(specs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return dataloader

def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            outputs = model(images)
            
            _, preds = torch.max(outputs, 1)
            st.write(preds)
            all_preds.extend(preds.cpu().numpy())

    return all_preds
