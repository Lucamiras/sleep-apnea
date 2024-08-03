import streamlit as st
import torch
import torchvision.models as models
from assets.texts import ANALYZE_INFO_TEXT
from src.globals import MEAN, STD, SIZE
from src.classifier import ApneaClassifier
from src.functions import *
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import librosa
import os
import numpy as np


st.header("Analyze your sleep")
st.sidebar.write("Upload a .wav recording of your sleep. The Deep Sleep analyzer will go through it and provide you with an estimated Apnea-Hypopnea Index (AHI) score.")
input_container = st.container(border=True)
results_container = st.container()
results_1, results_2, results_3 = results_container.columns(3)
playback_container = st.container()
playback_1, playback_2 = playback_container.columns(2)

file_uploader = input_container.file_uploader("Upload your recording here", type=["wav"])

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 3)
model.load_state_dict(torch.load("model/resnet50_state_dict.pth", map_location=torch.device('cpu')), strict=True)
model.eval()

if file_uploader:
    player = input_container.audio(file_uploader, format="audio/wav")
    analyze_button = input_container.button("Analyze!", use_container_width=True)
    input_container.markdown(ANALYZE_INFO_TEXT)
    
    # already chop file into segments
    audio, sr = load_audio(file_uploader)
    audio_chunks = split_audio(audio, sr)
    duration_in_minutes = np.round(len(audio) / sr / 60, 1)
    duration_in_hours = np.round(len(audio) / sr / 3600, 1)

    input_container.metric("Audio length (minutes)", duration_in_minutes)
    
    if len(os.listdir(".temp")) != 0:
        for file in os.listdir(".temp"):
            os.remove(os.path.join(".temp",file))
                      
    if len(os.listdir(".temp")) == 0:
        for i, chunk in enumerate(audio_chunks):
            generate_spectrogram(chunk, f".temp/output_{i}.png")
                      
    # Now we perform the classification
    dataloader = create_dataloader_from_chunks(".temp", MEAN, STD, SIZE)
    if analyze_button:
        with st.spinner("Analyzing..."):
            results, logits = run_inference(model, dataloader, "cpu")
            results_counter = sum([1 for result in results if result != 0])
            
            results_1.metric("Total apnea events detected", results_counter)
            results_2.metric("Total apnea events per hour", round(calculate_event_per_hour(results_counter, duration_in_hours), 1))
            results_3.write(f"AHI score:")
            results_3.subheader(get_ahi_score(results_counter, duration_in_hours))
            confidence = probabilities_from_logits(logits)
            plot_results_plotly(results, results_container, confidence)
            df = create_dataframe(results, confidence)
            results_container.dataframe(df)