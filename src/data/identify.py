from pathlib import Path
from classes import ImageRecord
import re

def get_metadata(
    file: Path,
    output: bool = False
) -> ImageRecord:
    file_name = file.name
    if file_name[0] == "f":
        data_type = "no_pathology"
        parent_name = file.parent.name
        match_parent = re.search(r"v_(?P<v>\d+)_s_(?P<s1>\d+)-(?P<s2>\d+)", parent_name)
        match = re.search(r"f_(?P<f>\d+)\.jpg", file_name)
        v = int(match_parent.group("v"))
        f = int(match.group("f"))
        s1 = int(match_parent.group("s1"))
        s2 = int(match_parent.group("s2"))
        if output:       
            divider = "=" * 45
            print(divider)
            print(" FILE METADATA ".center(45, "="))
            print(divider)
            print(f"  - Parent Name : {parent_name}")
            print(f"  - File Name : {file_name}")
            print(f"  - Type  : {data_type}")
            print(f"  - Video : {v}")
            print(f"  - Frame : {f}")
            print(f"\n{divider}")
        assert (s1 <= f < s2), f"Frame ID not in sequence range: {parent_name}/{file_name}"
        return ImageRecord(
            path = file,
            label = data_type,
            case_id = None,
            video_id = v,
            frame_id = f,
            sequence_start = s1,
            sequence_end = s2
        )
    elif file_name[0] == "c":
        data_type = "pathology"
        match = re.search(r"c_(?P<c>\d+)_v_\(video_(?P<video>\d+)\.mp4\)_f_(?P<f>\d+)\.jpg", file_name)
        c = int(match.group("c"))
        v = int(match.group("video"))
        f = int(match.group("f"))
        if output:       
            divider = "=" * 45
            print(divider)
            print(" FILE METADATA ".center(45, "="))
            print(divider)
            print(f"  - File Name : {file_name}")
            print(f"  - Type  : {data_type}")
            print(f"  - Case  : {c}")
            print(f"  - Video : {v}")
            print(f"  - Frame : {f}")
            print(f"\n{divider}")
        assert (c > 0), f"Case ID is invalid: {file_name}"
        assert (v > 0), f"Video ID is invalid: {file_name}"
        assert (f >= 0), f"Frame ID is negative: {file_name}"
        return ImageRecord(
            path = file,
            label = data_type,
            case_id = c,
            video_id = v,
            frame_id = f,
            sequence_start = None,
            sequence_end = None
        )
    else:
        raise ValueError("Unknown file classification.")