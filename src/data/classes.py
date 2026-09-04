from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImageRecord:
    path: Path
    label: str
    case_id: int | None
    video_id: int
    frame_id: int
    sequence_start: int | None
    sequence_end: int | None    