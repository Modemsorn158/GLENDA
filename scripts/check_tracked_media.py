"""Inspect the Git index for unapproved media and large blobs; do not inspect history."""
import hashlib
import io
import subprocess
import sys
from pathlib import PurePosixPath


MAX_BYTES = 5 * 1024 * 1024
APPROVED = {
    "reports/assets/composition.png": "f84828dee8b5c2d9065671cb56c5fe2eba1c7f72a1ab34dee3922100f596c050",
    "reports/assets/evaluation.png": "9d0fd99ee774b5ed2f68c148a1320642600f816e1e44a0ced201a07a87f97363",
}
BLOCKED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom",
                      ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".pdf", ".docx", ".xlsx",
                      ".zip", ".7z", ".tar", ".gz", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors",
                      ".npy", ".npz", ".pkl", ".pickle", ".h5", ".nii", ".nrrd", ".mha", ".mhd", ".svs"}


def check_index(repo="."):
    listing = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=repo)
    entries = []
    for item in listing.split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError("Unmerged entries, symlinks and submodules require separate review")
        entries.append((name.decode("utf-8"), oid))
    request = "".join(oid + "\n" for _, oid in entries).encode("ascii")
    descriptions = subprocess.check_output(["git", "cat-file", "--batch-check"], cwd=repo, input=request).decode("ascii").splitlines()
    if len(descriptions) != len(entries):
        raise ValueError("Cannot inspect indexed Git blob sizes")
    total = 0
    for (path, oid), description in zip(entries, descriptions):
        parts = description.split()
        if len(parts) != 3 or parts[0] != oid or parts[1] != "blob":
            raise ValueError("Cannot inspect an indexed Git blob")
        size = int(parts[2])
        if size > MAX_BYTES:
            raise ValueError(f"Tracked blob exceeds 5 MiB: {path}")
        total += size
    if total > 100 * 1024 * 1024:
        raise ValueError("Indexed blobs exceed the 100 MiB aggregate review limit")
    objects = subprocess.check_output(["git", "cat-file", "--batch"], cwd=repo, input=request)
    stream = io.BytesIO(objects)
    for path, oid in entries:
        header = stream.readline().decode("ascii").split()
        if len(header) != 3 or header[0] != oid or header[1] != "blob":
            raise ValueError("Cannot inspect an indexed Git blob")
        size = int(header[2])
        content = stream.read(size)
        if len(content) != size or stream.read(1) != b"\n":
            raise ValueError("Incomplete indexed Git blob")
        if size > MAX_BYTES:
            raise ValueError(f"Tracked blob exceeds 5 MiB: {path}")
        if path in APPROVED:
            if hashlib.sha256(content).hexdigest() != APPROVED[path]:
                raise ValueError(f"Approved figure changed and requires review: {path}")
            continue
        extension = PurePosixPath(path).suffix.lower()
        media_magic = (content.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"PK\x03\x04", b"%PDF-", b"II*\x00", b"MM\x00*")) or
                       content[128:132] == b"DICM" or (content[:4] == b"RIFF" and content[8:12] == b"WEBP") or
                       content[4:8] == b"ftyp")
        if extension in BLOCKED_EXTENSIONS or media_magic:
            raise ValueError(f"Unapproved media or archive in Git index: {path}")
    return len(entries)


def main():
    try:
        print(f"Checked {check_index()} indexed files; approved aggregate figures are hash-pinned.")
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
