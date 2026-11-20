# GLENDA research rebuild — early foundation

This is an **incomplete research rebuild**. It does not yet train or evaluate a model, establish a locked test set, or support clinical use. The original data are kept separately in `../read_only/raw` and are never modified by the audit.

## Dataset inventory

Install Python and Pillow, then run from this directory (PowerShell):

```powershell
$env:PYTHONPATH='src'
python -m glenda.inventory ..\read_only\raw outputs\audit
python -m unittest discover -s tests -v
```

The inventory checks positive images against the supplied COCO file, decodes every frame, records SHA-256 hashes, and reports exact duplicates and unreadable files. Derived files go in `outputs/`, which is ignored by Git. To screen cross-component near duplicates and create a development/test manifest, install the project dependencies (`python -m pip install -e .`), then run:

```powershell
python -m glenda.near_duplicates ..\read_only\raw outputs\audit\inventory.csv outputs\audit\near_duplicates.csv
python -m glenda.make_splits outputs\audit\inventory.csv outputs\audit\near_duplicates.csv outputs\splits\v1
```

The splitter refuses to overwrite a previous manifest. It joins case IDs, video IDs, identical image hashes, and all candidate pairs from the near-duplicate screen transitively. The latter is conservative: perceptual-hash matches may be false positives. The screen detects only pairs within Hamming distance three on a 64-bit difference hash; it does not prove absence of similar images. It creates an outer holdout using the first of five shuffled stratified-group folds and five development folds, without subsampling. It records source hashes, seed and class counts. No claim of patient-level independence is possible for negative videos with unknown cases. **A split must not be treated as a valid locked test if researchers have used its labels or images to make subsequent modeling choices.** The previous research used this dataset, so none of its partitions are external validation.

The official [GLENDA v1.5 documentation](https://ftp.itec.aau.at/datasets/GLENDA/) identifies annotated endometriosis frames and sequences with no visible endometriosis-related pathology as binary classification sources. The release is restricted to non-commercial scientific research and requires citation of Leibetseder et al., *GLENDA: Gynecologic Laparoscopy Endometriosis Dataset*, MMM 2020, [doi:10.1007/978-3-030-37734-2_36](https://doi.org/10.1007/978-3-030-37734-2_36).

## Critical limitations

Case IDs are present in positive filenames. Some negative videos match positive videos and inherit their known case ID. The remaining negative videos lack a case mapping, so grouping by available case/video IDs cannot guarantee patient independence. Seek authoritative case metadata before interpreting patient-level generalization. Exact-duplicate checks alone do not rule out near-duplicate leakage. The dataset's curated frame prevalence is not clinical prevalence. Prior reported scores remain unverified, and no test results are claimed here.
