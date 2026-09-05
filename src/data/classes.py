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
    
@dataclass
class Case:
    id: int
    paths: Path
    
@dataclass
class Video:
    id: int
    paths: Path
    
@dataclass
class Groups:
    cases: list[Case]
    videos: list[Video]