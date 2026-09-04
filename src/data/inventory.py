from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import statistics
from identify import get_metadata

root = Path(__file__).parent.parent.parent
path_pathology_frames = root / "data" / "raw" / "pathology" / "frames"
path_non_pathology_frames = root / "data" / "raw" / "no_pathology" / "frames"

def print_counter(
    counter: Counter
):
    return ", ".join(f"{k}: {v}" for k, v in counter.items())

if __name__ == "__main__":
    image_sizes = []
    image_color_modes = []
    records = []
    
    # Pathology
    count_pathology = 0
    pathology_file_type = []
    for file in path_pathology_frames.rglob('*'):
        if file.is_file():
            pathology_file_type.append(file.suffix)
            with Image.open(file) as image:
                count_pathology = count_pathology + 1
                width, height = image.size
                size = (width, height)
                color_mode = image.mode
                image_sizes.append(size)
                image_color_modes.append(color_mode)
                records.append(get_metadata(file))
    
    # No pathology
    count_no_pathology = 0
    no_pathology_file_type = []
    for file in path_non_pathology_frames.rglob('*'):
        if file.is_file():
            no_pathology_file_type.append(file.suffix)
            with Image.open(file) as image:
                count_no_pathology = count_no_pathology + 1
                width, height = image.size
                size = (width, height)
                color_mode = image.mode
                image_sizes.append(size)
                image_color_modes.append(color_mode)
                records.append(get_metadata(file))

    pathology_cases = set()
    pathology_videos = set()
    no_pathology_videos = set()
    case_to_videos = defaultdict(set)
    video_to_frames = defaultdict(set)
    sequence_frames = defaultdict(list)
    for r in records:
        f_id = int(r.frame_id)
        video_to_frames[r.video_id].add(f_id)
        if r.case_id is not None:
            pathology_cases.add(r.case_id)
            pathology_videos.add(r.video_id)
            case_to_videos[r.case_id].add(r.video_id)
        else:
            no_pathology_videos.add(r.video_id)
            seq_key = (r.video_id, int(r.sequence_start), int(r.sequence_end))
            sequence_frames[seq_key].append(f_id)
    overlap_videos = pathology_videos & no_pathology_videos
    videos_per_case = [len(vids) for vids in case_to_videos.values()]
    min_v_case = min(videos_per_case) if videos_per_case else 0
    max_v_case = max(videos_per_case) if videos_per_case else 0
    mean_v_case = statistics.mean(videos_per_case) if videos_per_case else 0
    median_v_case = statistics.median(videos_per_case) if videos_per_case else 0

    divider = "=" * 45
    print(divider)
    print(" SUMMARY REPORT ".center(45, "="))
    print(divider)
    print("\n[ Pathology ]")
    print(f"  - Image Count : {count_pathology}")
    print(f"  - File Types  : {print_counter(Counter(pathology_file_type))}")
    print("\n[ No Pathology ]")
    print(f"  - Image Count : {count_no_pathology}")
    print(f"  - File Types  : {print_counter(Counter(no_pathology_file_type))}")
    print("\n[ Global Image Metadata ]")
    print(f"  - Image Sizes : {print_counter(Counter(image_sizes))}")
    print(f"  - Color Modes : {print_counter(Counter(image_color_modes))}")
    print(f"\nTotal records: {len(records)}")
    print(f"Pathology records: {count_pathology}")
    print(f"No-pathology records: {count_no_pathology}")
    print(f"\nUnique pathology cases: {len(pathology_cases)}")
    print(f"Unique pathology videos: {len(pathology_videos)}")
    print(f"Unique no-pathology videos: {len(no_pathology_videos)}")
    print("\nVideos present in both classes:")
    if overlap_videos:
        for vid in sorted(overlap_videos):
            print(f"    - {vid}")
    else:
        print("    None")
    print("\nNumber of videos per pathology case:")
    print(f"    min: {min_v_case}")
    print(f"    max: {max_v_case}")
    print(f"    mean: {mean_v_case:.2f}")
    print(f"    median: {median_v_case}")
    print("\nNumber of frames per video:")
    for vid, frames in sorted(video_to_frames.items()):
        print(f"    Video {vid}: {len(frames)} frames")
    print("\nNo-Pathology Sequence Bounds Validation:")
    for (vid, seq_start, seq_end), frames in sorted(sequence_frames.items()):
        min_f = min(frames)
        max_f = max(frames)
        start_valid = min_f == seq_start
        end_valid = max_f == (seq_end - 1)
        status = "OK" if (start_valid and end_valid) else "FAILED"
        print(
            f"  Video {vid} (Seq {seq_start}-{seq_end}): [{status}] "
            f"min(frame_id)={min_f} == seq_start ({start_valid}), "
            f"max(frame_id)={max_f} == (seq_end - 1) ({end_valid})"
        )
    print(f"\n{divider}")