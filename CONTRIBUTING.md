# Contributing to GLENDA

This is a research-only project. Do not commit patient images, derived image crops, predictions containing identifiers, datasets, checkpoints, credentials or other restricted files. The current checkout contains two reviewed aggregate figures; their exact bytes are permitted, not arbitrary replacements.

Before committing, review the staged files and run:

```text
python scripts/check_tracked_media.py
python -m unittest discover -s tests -v
```

The media check inspects the Git index, so stage your changes before running it. It blocks selected image, medical-image, video, archive, document and model formats, some recognizable file signatures, blobs over 5 MiB, and more than 100 MiB of indexed blobs in total. The two approved aggregate figures are pinned by SHA-256. CI runs the same check. A new or changed figure needs explicit review and an intentional update to the allowlist; never use an exception for patient data.

This is a guardrail, not a privacy scanner or access-control boundary: it cannot identify all sensitive text, disguised encodings, file formats, Git LFS content, remote caches, historical objects or files changed after the check. Review the actual staged diff and follow the data custodian's policy. The check is not installed automatically as a Git hook; run it before committing. The Git history already contains dataset images; see [image handling](docs/data_privacy.md).

Use synthetic data for tests. Do not reuse the consumed holdout for model, threshold or calibration decisions. Explain whether reported results are development-only and list known patient/group independence limits. For the operational workflow see the [technical guide](docs/technical-guide.md).
