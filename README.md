# GLENDA research rebuild — early foundation

This is an **incomplete research rebuild**. It does not yet train or evaluate a model, establish a locked test set, or support clinical use. The original data are kept separately in `../read_only/raw` and are never modified by the audit.

## Dataset inventory

Install Python and Pillow, then run from this directory (PowerShell):

```powershell
$env:PYTHONPATH='src'
python -m glenda.inventory ..\read_only\raw outputs\audit
python -m unittest discover -s tests -v
```

The tested GPU environment uses Windows x64, Python 3.14.7 and CUDA-enabled PyTorch 2.15.0+cu132. Its installed package versions are recorded in `requirements-gpu.lock`; install them in a fresh virtual environment with `python -m pip install -r requirements-gpu.lock --extra-index-url https://download.pytorch.org/whl/cu132`. CUDA availability depends on the local NVIDIA driver. The lock records versions, not package hashes or a guarantee of bitwise reproducibility; other platforms may require different builds. A CPU environment is sufficient for smoke tests.

The inventory checks positive images against the supplied COCO file, decodes every frame, records SHA-256 hashes, and reports exact duplicates and unreadable files. Derived files go in `outputs/`, which is ignored by Git. To screen cross-component near duplicates and create a development/test manifest, install the project dependencies (`python -m pip install -e .`), then run:

```powershell
python -m glenda.near_duplicates ..\read_only\raw outputs\audit\inventory.csv outputs\audit\near_duplicates.csv
python -m glenda.exploration outputs\audit\inventory.csv outputs\audit\composition.png
python -m glenda.make_splits outputs\audit\inventory.csv outputs\audit\near_duplicates.csv outputs\splits\v1
```

The splitter refuses to overwrite a previous manifest. It joins case IDs, video IDs, identical image hashes, and all candidate pairs from the near-duplicate screen transitively. The latter is conservative: perceptual-hash matches may be false positives. The screen detects only pairs within Hamming distance three on a 64-bit difference hash; it does not prove absence of similar images. It creates an outer holdout using the first of five shuffled stratified-group folds and five development folds, without subsampling. It records source hashes, seed and class counts. No claim of patient-level independence is possible for negative videos with unknown cases. **A split must not be treated as a valid locked test if researchers have used its labels or images to make subsequent modeling choices.** The previous research used this dataset, so none of its partitions are external validation.

The official [GLENDA v1.5 documentation](https://ftp.itec.aau.at/datasets/GLENDA/) identifies annotated endometriosis frames and sequences with no visible endometriosis-related pathology as binary classification sources. The release is restricted to non-commercial scientific research and requires citation of Leibetseder et al., *GLENDA: Gynecologic Laparoscopy Endometriosis Dataset*, MMM 2020, [doi:10.1007/978-3-030-37734-2_36](https://doi.org/10.1007/978-3-030-37734-2_36).

## Critical limitations

Case IDs are present in positive filenames. Some negative videos match positive videos and inherit their known case ID. The remaining negative videos lack a case mapping, so grouping by available case/video IDs cannot guarantee patient independence. Seek authoritative case metadata before interpreting patient-level generalization. Exact-duplicate checks alone do not rule out near-duplicate leakage. The dataset's curated frame prevalence is not clinical prevalence. Prior reported scores remain unverified, and no test results are claimed here.

## Development-fold training (experimental)

Install the optional training dependencies with `python -m pip install -e '.[training]'`. A development fold can be run with `python -m glenda.train_fold ..\read_only\raw outputs\splits\v1\manifest.csv outputs\runs\fold0 --fold 0`. Defaults use ImageNet EfficientNet-B0 with frozen features, AdamW and validation average precision for early stopping. `--fine-tuning partial` or `full`, `--weighted`, and `--architecture resnet50` change the comparison. Run all folds for a configuration before comparing it. Pretrained weights may require a download. Inputs are resized to 224 by 224 (changing aspect ratio), with a horizontal flip during training and ImageNet normalization. This is a starting transformation, not an established clinically optimal preprocessing policy.

The command never evaluates the test partition. It records configuration, manifest hash, epoch losses, validation average precision and the best checkpoint, and refuses to overwrite an existing run. A small prespecified comparison can be started with `python -m glenda.cross_validate ..\read_only\raw outputs\splits\v1\manifest.csv configs\pilot.json outputs\search\pilot`. This runs five folds per configuration and records fold failures; rerunning the command skips completed or failed folds. New searches record training-source hashes and stop if the code changes between folds. The earlier feasibility runs predate this check. Interrupted and failed folds are retained for inspection rather than silently retried. The configurations compare frozen and partial fine-tuning, class-weighted loss, and a frozen ResNet50 baseline. No longer pilot results are yet available.

After all tuning folds complete, `python -m glenda.select_candidate outputs\search\tuning` freezes the EfficientNet trial with the highest mean development AP. Then use `python -m glenda.select_threshold outputs\splits\v1\manifest.csv outputs\search\tuning\trial_000 outputs\search\tuning\selected_threshold.json`, substituting the selected trial directory. It selects a pooled out-of-fold F1 threshold and checks fold completion, exact development coverage, labels, groups and manifest identity. Threshold selection on the same predictions is optimistic, and must not be reported as independent performance. Fold scores also drive early stopping and model selection, so they can be optimistic.

After model and threshold selection, `python -m glenda.fit_final ..\read_only\raw outputs\splits\v1\manifest.csv outputs\search\tuning\trial_000 outputs\search\tuning\selected_threshold.json outputs\final\selected` fits on all development frames; substitute the selected trial directory. It uses the median of the five validation-selected best epoch counts, fixes the selected threshold, verifies fold configurations and prediction hashes, and never opens test images. The final fit has been tested on synthetic data but has not been run for this study.

Only after the model, fixed epoch count and threshold are selected and final fitting is complete, run `python -m glenda.evaluate_test ..\read_only\raw outputs\splits\v1\manifest.csv outputs\final\selected\final.pt outputs\test\selected outputs\test\ledger`. The command creates a manifest-specific ledger entry before reading test images, and refuses a second evaluation with the same ledger. Keep that ledger for every evaluation of this manifest; this is a workflow safeguard, not a security barrier. An interrupted run leaves the entry marked `started`; investigate and document the reason before any retry. The output includes predictions, classification metrics, Brier score, fixed-threshold sensitivity, cluster-bootstrap intervals, ROC, precision-recall and calibration plots, plus a local contact sheet of up to eight highest-scoring false positives and eight lowest-scoring false negatives. That sheet contains derived medical images and must remain private. Bootstrap clusters reflect only known linked groups, so their intervals cannot account for unidentified patient overlap. Do not use test sensitivity analyses to change the threshold.

After successful completion, `python -m glenda.summarize_test outputs\test\selected outputs\test\ledger reports\test_results.md` generates a de-identified result table from the recorded metrics and verifies the ledger. The full prediction and image files stay in the ignored `outputs/` directory.

For a research-only demonstration after a selected final model has been evaluated, install `python -m pip install -e '.[training,demo]'` and run `python -m glenda.demo outputs\final\selected\final.pt`. It listens on localhost, displays the fixed threshold and an uncalibrated score, and is not for diagnosis. The prediction function has a unit test, but the Gradio interface has not yet been exercised with a final model.

The project does not yet implement epoch-level checkpoint resumption. GPU kernels are not guaranteed deterministic. Do not present smoke tests as research results. The held-out evaluation has not been run.
