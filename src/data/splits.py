from collections import defaultdict
from classes import ImageRecord, Groups, Case, Video

def split_record(records: list[ImageRecord]) -> Groups:
    case_paths = defaultdict(list)
    video_paths = defaultdict(list)

    for record in records:
        if record.case_id:
            case_paths[record.case_id].append(record.path)
        video_paths[record.video_id].append(record.path)

    cases = [Case(id=case_id, paths=paths) for case_id, paths in case_paths.items()]
    videos = [Video(id=video_id, paths=paths) for video_id, paths in video_paths.items()]

    return Groups(cases=cases, videos=videos)