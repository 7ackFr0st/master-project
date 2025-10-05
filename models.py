# models.py
from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict

@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str
    translation: Optional[str] = None

@dataclass
class Transcript:
    segments: List[Segment]

    def as_dict(self):
        return {i: seg for i, seg in enumerate(self.segments)}

    def by_speaker(self):
        d = defaultdict(list)
        for seg in self.segments:
            d[seg.speaker].append(seg)
        return d
