"""Record and verify result hashes; the ledger must itself be access-controlled."""
import hashlib
from pathlib import Path


def result_hashes(output):
    output = output.resolve()
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Evaluation output must contain regular files only")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = digest.hexdigest()
    if not {"metrics.json", "predictions.csv", "group_errors.csv", "evaluation.png"} <= files.keys():
        raise ValueError("Evaluation is missing required outputs")
    return files


def verify_results(output: Path, lock: dict, *, allow_legacy_unverified=False):
    expected = lock.get("artifact_sha256")
    if expected is None:
        if allow_legacy_unverified:
            return
        raise ValueError("Legacy ledger has no result hashes; explicit legacy acknowledgement required")
    if not isinstance(expected, dict) or result_hashes(output) != expected:
        raise ValueError("Evaluation output hashes do not match completed ledger")
