import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import numpy as np
import streamlit as st
from src.dataset import SpectrogramDataset
from src.globals import TARGET_SR, CHUNK_DURATION, AHI, CLASSES
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import plotly.graph_objects as go
import pandas as pd
import time


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

            logits = model(images)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds, logits

def probabilities_from_logits(logits):
    probs = f.softmax(logits)
    return [torch.max(p).item() for p in probs]

def calculate_event_per_hour(count, duration):
    if duration == 0:
        return 0
    return count / duration

def get_ahi_score(count, duration, ahi_index=AHI):
    print("AHI score called")
    events_per_hour = calculate_event_per_hour(count, duration)
    if events_per_hour > 120:
        return "Severe"
    for ahi_range, ahi_score in ahi_index.items():
        if int(events_per_hour) in ahi_range:
            return ahi_score

def plot_results_plotly(values, streamlit_container, confidence):
    colors = ['white', 'pink', 'purple']
    bar_colors = [colors[val] for val in values]
    fig = go.Figure()

    for idx, (val, height) in enumerate(zip(values, confidence)):
        fig.add_trace(go.Bar(
            x=[idx],
            y=[height] if val > 0 else [0],
            marker=dict(color=bar_colors[idx]),
            showlegend=False,
            text=str(val),
            textposition='inside' if val > 0 else 'none',
            textfont=dict(color='white' if val > 0 else 'black')
        ))

    fig.update_layout(
        title='Detected events over time',
        xaxis=dict(
            tickvals=list(range(0, len(values), 10)),
            ticktext=[str(i/2) for i in range(0, len(values), 10)]
        ),
        yaxis=dict(range=[0, 1], title='Height'),
        xaxis_title='Index',
        yaxis_title='Height',
        barmode='stack'
    )

    return streamlit_container.plotly_chart(fig)

def create_playback_snippets(results, chunk_duration=CHUNK_DURATION, classes=CLASSES):
    labels = [classes[result] for result in results]
    list_of_snippets = [f"{label} at {i * chunk_duration}" for i, label in enumerate(labels)]
    return list_of_snippets

def create_dataframe(results, confidence, chunk_duration=CHUNK_DURATION, classes=CLASSES):
    df = pd.DataFrame()
    df['time'] = [i * chunk_duration for i in range(len(results))]
    df['event'] = [classes[result] for result in results]
    df['confidence'] = [np.round(conf,2) for conf in confidence]
    return df

def render_progress_bar(duration_in_minutes, streamlit_container):
    complete = False
    expected_number_of_chunks = int(duration_in_minutes * 2) # 120
    progress_text = "Transforming audio ... "
    complete_text = "Done!"
    my_bar = streamlit_container.progress(0, text=progress_text)
    while not complete:
        time.sleep(0.01)
        progress = len(os.listdir('.temp'))
        progress_percent = int((100 / expected_number_of_chunks) * progress)
        my_bar.progress(progress_percent, text=progress_text)
        if progress == expected_number_of_chunks:
            complete = True
    my_bar.progress(100, text=complete_text)
    time.sleep(1)
    my_bar.empty()

    


        

