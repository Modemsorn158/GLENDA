from pathlib import Path
import re

def get_metadata(
    file: Path
) -> None:
    file_name = file.name
    if file_name[0] == "f":
        data_type = "no_pathology"
        parent_name = file.parent.name
        match_parent = re.search(r"v_(?P<v>\d+)_s_(?P<s1>\d+)-(?P<s2>\d+)", parent_name)
        match = re.search(r"f_(?P<f>\d+).jpg", file_name)
        v = match_parent.group("v")
        f = match.group("f")       
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
    elif file_name[0] == "c":
        data_type = "pathology"
        match = re.search(r"c_(?P<c>\d+)_v_\(video_(?P<video>\d+)\.mp4\)_f_(?P<f>\d+)\.jpg", file_name)
        c = match.group("c")
        v = match.group("video")
        f = match.group("f")       
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
    else:
        raise ValueError("Unknown file classification.")