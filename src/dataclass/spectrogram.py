import numpy as np
from dataclasses import dataclass


@dataclass
class Spectrogram:
    label: str
    patient_id: str
    data: np.ndarray