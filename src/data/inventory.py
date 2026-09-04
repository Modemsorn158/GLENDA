from pathlib import Path
from PIL import Image
from collections import Counter

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
    print(f"\n{divider}")