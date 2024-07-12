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
import os
import plotly.graph_objects as go


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
            all_preds.extend(preds.cpu().numpy())

    return all_preds

def calculate_event_per_hour(count, duration):
    if duration == 0:
        return 0
    return count / duration

def get_ahi_score(count, duration):
    events_per_hour = calculate_event_per_hour(count, duration)
    match events_per_hour:
        case events_per_hour if events_per_hour in range(5):
            return "None"
        case events_per_hour if events_per_hour in range(5, 15):
            return "Mild"
        case events_per_hour if events_per_hour in range(15, 30):
            return "Moderate"  
        case events_per_hour if events_per_hour > 30:
            return "Severe"

def plot_results_plotly(values, streamlit_container, duration):
    colors = ['pink', 'purple']
    labels = ['Hypopnea', 'Obstructive Apnea']

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[idx for idx, val in enumerate(values) if val == 1],
        y=[1] * values.count(1),
        marker=dict(color=colors[0]),
        name=labels[0],
        text=['1'] * values.count(1),
        textposition='inside',
        textfont=dict(color='white')
    ))

    fig.add_trace(go.Bar(
        x=[idx for idx, val in enumerate(values) if val == 2],
        y=[1] * values.count(2),
        marker=dict(color=colors[1]),
        name=labels[1],
        text=['2'] * values.count(2),
        textposition='inside',
        textfont=dict(color='white')
    ))

    fig.update_layout(
        title='Bar Chart with Same Height and Different Colors',
        xaxis=dict(
            tickvals=list(range(0, duration, 5)),
            ticktext=[str(i) for i in range(0, duration, 5)]
        ),
        xaxis_title='Time',
        barmode='stack',
        showlegend=True
    )


    return streamlit_container.plotly_chart(fig)
