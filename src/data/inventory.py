from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import statistics
from identify import get_metadata

root = Path(__file__).resolve().parent.parent.parent
path_pathology_frames = root / "data" / "raw" / "pathology" / "frames"
path_non_pathology_frames = root / "data" / "raw" / "no_pathology" / "frames"

def print_counter(counter: Counter) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counter.items())

def process_directory(directory_path: Path) -> tuple[list, list, list, list]:
    file_types = []
    records = []
    sizes = []
    color_modes = []

    for file in directory_path.rglob("*"):
        if file.is_file():
            file_types.append(file.suffix)
            with Image.open(file) as img:
                sizes.append(img.size)
                color_modes.append(img.mode)
            records.append(get_metadata(file))
            
    return file_types, sizes, color_modes, records

if __name__ == "__main__":
    pathology_types, path_sizes, path_modes, path_records = process_directory(path_pathology_frames)
    no_pathology_types, no_path_sizes, no_path_modes, no_path_records = process_directory(path_non_pathology_frames)
    count_pathology = len(path_records)
    count_no_pathology = len(no_path_records)
    records = path_records + no_path_records
    image_sizes = path_sizes + no_path_sizes
    image_color_modes = path_modes + no_path_modes
    pathology_cases = set()
    pathology_videos = set()
    no_pathology_videos = set()
    case_to_videos = defaultdict(set)
    video_to_frames = defaultdict(set)
    sequence_frames = defaultdict(list)

    for r in records:
        video_to_frames[r.video_id].add(r.frame_id)

        if r.case_id is not None:
            pathology_cases.add(r.case_id)
            pathology_videos.add(r.video_id)
            case_to_videos[r.case_id].add(r.video_id)
        else:
            no_pathology_videos.add(r.video_id)
            if r.sequence_start is not None and r.sequence_end is not None:
                seq_key = (r.video_id, r.sequence_start, r.sequence_end)
                sequence_frames[seq_key].append(r.frame_id)

    overlap_videos = pathology_videos & no_pathology_videos
    videos_per_case = [len(vids) for vids in case_to_videos.values()]
    min_v_case = min(videos_per_case) if videos_per_case else 0
    max_v_case = max(videos_per_case) if videos_per_case else 0
    mean_v_case = statistics.mean(videos_per_case) if videos_per_case else 0
    median_v_case = statistics.median(videos_per_case) if videos_per_case else 0
    video_to_records = defaultdict(list)
    case_to_records = defaultdict(list)
    all_videos = set()

    for r in records:
        video_to_records[r.video_id].append(r)
        all_videos.add(r.video_id)
        if r.case_id is not None:
            case_to_records[r.case_id].append(r)
            
    parent = {}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
    for cid in pathology_cases:
        parent[f"c_{cid}"] = f"c_{cid}"
    for vid in all_videos:
        parent[f"v_{vid}"] = f"v_{vid}"

    for r in records:
        if r.case_id is not None:
            union(f"c_{r.case_id}", f"v_{r.video_id}")

    group_clusters = defaultdict(list)
    group_cases = defaultdict(set)
    group_videos = defaultdict(set)

    for r in records:
        node_key = f"c_{r.case_id}" if r.case_id is not None else f"v_{r.video_id}"
        root_node = find(node_key)
        group_clusters[root_node].append(r)
        group_videos[root_node].add(r.video_id)
        if r.case_id is not None:
            group_cases[root_node].add(r.case_id)

    total_groups = len(group_clusters)
    case_backed_groups = 0
    video_only_groups = 0
    frames_per_group = []
    pathology_only_groups = 0
    no_pathology_only_groups = 0
    mixed_groups = 0
    record_group_map = defaultdict(int)
    case_group_map = defaultdict(int)
    video_group_map = defaultdict(int)

    for group_idx, (root_node, group_recs) in enumerate(group_clusters.items(), start=1):
        has_cases = len(group_cases[root_node]) > 0
        if has_cases:
            case_backed_groups += 1
        else:
            video_only_groups += 1

        frames_per_group.append(len(group_recs))

        labels = {r.label for r in group_recs}
        if labels == {"pathology"}:
            pathology_only_groups += 1
        elif labels == {"no_pathology"}:
            no_pathology_only_groups += 1
        else:
            mixed_groups += 1

        for r in group_recs:
            record_group_map[r.path] += 1
        for cid in group_cases[root_node]:
            case_group_map[cid] += 1
        for vid in group_videos[root_node]:
            video_group_map[vid] += 1

    min_fg = min(frames_per_group) if frames_per_group else 0
    max_fg = max(frames_per_group) if frames_per_group else 0
    mean_fg = statistics.mean(frames_per_group) if frames_per_group else 0
    median_fg = statistics.median(frames_per_group) if frames_per_group else 0

    assert len(record_group_map) == len(records) and all(count == 1 for count in record_group_map.values()), \
        "Verification Failed: Not every record occurs in exactly one group."
    
    assert len(case_group_map) == len(pathology_cases) and all(count == 1 for count in case_group_map.values()), \
        "Verification Failed: Not every pathology case occurs in exactly one group."
    
    assert len(video_group_map) == len(all_videos) and all(count == 1 for count in video_group_map.values()), \
        "Verification Failed: Not every video occurs in exactly one group."

    divider = "=" * 45
    print(divider)
    print(" SUMMARY REPORT ".center(45, "="))
    print(divider)
    print("\n[ Pathology ]")
    print(f"  - Image Count : {count_pathology}")
    print(f"  - File Types  : {print_counter(Counter(pathology_types))}")
    print("\n[ No Pathology ]")
    print(f"  - Image Count : {count_no_pathology}")
    print(f"  - File Types  : {print_counter(Counter(no_pathology_types))}")
    print("\n[ Global Image Metadata ]")
    print(f"  - Image Sizes : {print_counter(Counter(image_sizes))}")
    print(f"  - Color Modes : {print_counter(Counter(image_color_modes))}")
    print(f"\nTotal records: {len(records)}")
    print(f"Pathology records: {count_pathology}")
    print(f"No-pathology records: {count_no_pathology}")
    print(f"\nUnique pathology cases: {len(pathology_cases)}")
    print(f"Unique pathology videos: {len(pathology_videos)}")
    print(f"Unique no-pathology videos: {len(no_pathology_videos)}")

    print("\n[ Split Groups Analysis ]")
    print(f"Total split groups: {total_groups}")
    print(f"Case-backed groups: {case_backed_groups}")
    print(f"Video-only groups: {video_only_groups}")
    print("\nFrames per group:")
    print(f"    min: {min_fg}")
    print(f"    max: {max_fg}")
    print(f"    mean: {mean_fg:.2f}")
    print(f"    median: {median_fg}")
    print("\nClass composition per group:")
    print(f"    pathology only: {pathology_only_groups}")
    print(f"    no-pathology only: {no_pathology_only_groups}")
    print(f"    mixed: {mixed_groups}")

    print("\n[ Verification ]")
    print("  [OK] Every record occurs in exactly one group.")
    print("  [OK] Every pathology case occurs in exactly one group.")
    print("  [OK] Every video occurs in exactly one group.")
    print(f"\n{divider}")