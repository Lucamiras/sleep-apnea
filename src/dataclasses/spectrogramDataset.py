from dataclasses import dataclass, field
from typing import Dict, List
from src.dataclasses.spectrogram import Spectrogram


@dataclass(frozen=True)
class SpectrogramDataset:
    clip_length: float = field()
    data: Dict[str, List[Spectrogram]] = field()

    def __repr__(self):
        return (f"SpectrogramDataset(clip_length={self.clip_length}, "
                f"train={len(self.data.get('train'))}, "
                f"val={len(self.data.get('val'))}, "
                f"test={len(self.data.get('test'))}. "
                "Each dataset contains a class label (str), the signal data (np.ndarray), and a patient_id (str).")