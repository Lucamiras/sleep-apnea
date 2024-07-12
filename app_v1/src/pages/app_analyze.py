import librosa
import streamlit as st
from app_v1.src.functions import classify_chunk, count_classifications, plot_classifications, get_ahi


# Placeholder for my classifier
def main():
    st.header("Upload a file for analysis")
    st.write("File must be wav format")

    input_container = st.container(border=True)
    # Upload audio file
    uploaded_file = input_container.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file:
        # Load audio file
        audio, sr = librosa.load(uploaded_file)
        duration = librosa.get_duration(y=audio, sr=sr)
        input_container.audio(uploaded_file, format='audio/wav')

        chunk_duration = input_container.number_input('Chunk duration (seconds)', min_value=1, max_value=int(duration), value=5)
        n_chunks = int(duration // chunk_duration)

        classifications = []

        for i in range(n_chunks):
            start = int(i * chunk_duration * sr)
            end = int((i + 1) * chunk_duration * sr)
            chunk = audio[start:end]
            classification = classify_chunk(chunk)
            classifications.append(classification)

        diag_box = st.container()
        plot_classifications(classifications, chunk_duration)
        num_classifications = count_classifications(classifications)
        diagnosis = get_ahi(num_classifications)

        with diag_box:
            st.header("Results")
            col1, col2 = st.columns(2)
            col1.metric("Number of events", num_classifications, delta=-1)
            col2.metric("Apnea Hypopnea Index", diagnosis, delta=4)

        st.write("Classifications:", classifications)

main()



