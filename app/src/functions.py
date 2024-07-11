import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def classify_chunk(chunk):
    classes = ["Apnea", "Hypopnea", "ObstructiveApnea", "NoApnea"]
    return np.random.choice(classes, p=[0.1, 0.1, 0.1, 0.7])  # Replace with actual model prediction


def count_classifications(classifications) -> int:
    """
    This function returns the number of positive classifications.
    """
    return len([event for event in classifications if event != 'NoApnea'])


def plot_classifications(classifications, chunk_duration):
    fig, ax = plt.subplots(figsize=(10, 2))

    start_times = np.arange(0, len(classifications) * chunk_duration, chunk_duration)
    colors = {'Apnea': 'yellow', 'Hypopnea': 'orange', 'ObstructiveApnea': 'red', 'NoApnea': 'white'}

    # add a new color patch for each classification
    for i, classification in enumerate(classifications):
        if classification != 'NoApnea':
            ax.add_patch(plt.Rectangle((start_times[i], 0), chunk_duration, 1, color=colors[classification], alpha=0.6))

    # This is the legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values() if color != 'white']
    labels = [label for label in colors.keys() if label != 'NoApnea']
    ax.legend(handles, labels, loc='upper right')

    # this is the plot
    ax.set_xlim(0, start_times[-1] + chunk_duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    st.pyplot(fig)


def get_ahi(num_classifications):
    ahi = {
        'No Apnea': range(0, 5),
        'Mild apnea': range(5, 15),
        'Medium apnea': range(15, 30),
        'Severe apnea': range(30, 10_000)
    }
    diagnosis = None
    for ahi_label, ahi_range in ahi.items():
        if num_classifications in ahi_range:
            diagnosis = ahi_label

    return diagnosis
