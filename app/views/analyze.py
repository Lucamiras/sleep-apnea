import streamlit as st
import torch
from assets.texts import ANALYZE_INFO_TEXT
from src.globals import MEAN, STD, SIZE, TARGET_SR
from src.classifier import ApneaClassifier
from src.functions import load_audio, split_audio, generate_spectrogram, create_dataloader_from_chunks, run_inference
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import librosa
import os


st.title("Analyze your sleep")

input_container = st.container(border=True)
results_container = st.container()

file_uploader = input_container.file_uploader("Upload your sleep data", type=["wav"])

model = ApneaClassifier(num_classes=3)
model.load_state_dict(torch.load("model/resnet50_state_dict.pth"), strict=False)
model.eval()

if file_uploader:
    player = input_container.audio(file_uploader.name, format=file_uploader.type)
    analyze_button = input_container.button("Analyze!", use_container_width=True)
    input_container.markdown(ANALYZE_INFO_TEXT)
    
    # already chop file into segments
    audio, sr = load_audio(file_uploader)
    st.text(f"Audio length: {len(audio) / sr:.2f} seconds")
    audio_chunks = split_audio(audio, sr)
    st.text(f"Number of chunks: {len(audio_chunks)}")
    st.text(audio_chunks[0].shape)
    if len(os.listdir(".temp")) == 0:
        for i, chunk in enumerate(audio_chunks):
            generate_spectrogram(chunk, f".temp/output_{i}.png")
    # Now we perform the classification
    dataloader = create_dataloader_from_chunks(".temp", MEAN, STD, SIZE)
    if analyze_button:
        with st.spinner("Analyzing..."):
            results = run_inference(model, dataloader, "cpu")
            #results_container.write(results)
            st.bar_chart(results)