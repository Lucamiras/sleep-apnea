from dataclasses import dataclass, field, asdict
from typing import List, Union
from numpy import ndarray


@dataclass
class ApneaEvent:
    start: float = field()
    end: float = field()
    label: str = field()
    signal: List[Union[ndarray, None]] = field()

@dataclass(frozen=True)
class ApneaEvents:
    acq_number: str = field()
    gender: str = field()
    events: List[ApneaEvent] = field()

    def to_dict(self):
        return asdict(self)