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

## Development-fold training (experimental)

Install the optional training dependencies with `python -m pip install -e '.[training]'`. A development fold can be run with `python -m glenda.train_fold ..\read_only\raw outputs\splits\v1\manifest.csv outputs\runs\fold0 --fold 0`. Defaults use ImageNet EfficientNet-B0 with frozen features, AdamW and validation average precision for early stopping. `--fine-tuning partial` or `full`, `--weighted`, and `--architecture resnet50` change the comparison. Run all folds for a configuration before comparing it. Pretrained weights may require a download. Inputs are resized to 224 by 224 (changing aspect ratio), with a horizontal flip during training and ImageNet normalization. This is a starting transformation, not an established clinically optimal preprocessing policy.

The command never evaluates the test partition. It records configuration, manifest hash, epoch losses, validation average precision and the best checkpoint, and refuses to overwrite an existing run. A small prespecified comparison can be started with `python -m glenda.cross_validate ..\read_only\raw outputs\splits\v1\manifest.csv configs\pilot.json outputs\search\pilot`. This runs five folds per configuration and records fold failures; rerunning the command skips completed or failed folds. Interrupted and failed folds are retained for inspection rather than silently retried. The configurations compare frozen and partial fine-tuning, class-weighted loss, and a frozen ResNet50 baseline. No pilot results are yet available.

The project does not yet implement out-of-fold threshold selection, final-model fitting, calibration, uncertainty, test evaluation, epoch-level checkpoint resumption, or a demo. Validation folds also drive early stopping, so the resulting fold scores can be optimistic. GPU kernels are not guaranteed deterministic. Do not present smoke tests as research results.
