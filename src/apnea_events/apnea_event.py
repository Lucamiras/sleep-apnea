import numpy as np
from dataclasses import dataclass

@dataclass
class ApneaEvent:
    index: int
    acq_number: str
    start: float
    end: float
    label: str
    signal: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)