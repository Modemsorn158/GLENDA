# GLENDA research rebuild — early foundation

This is an **incomplete research rebuild**. It does not yet train or evaluate a model, establish a locked test set, or support clinical use. The original data are kept separately in `../read_only/raw` and are never modified by the audit.

## Dataset inventory

Install Python and Pillow, then run from this directory (PowerShell):

```powershell
$env:PYTHONPATH='src'
python -m glenda.inventory ..\read_only\raw outputs\audit
python -m unittest discover -s tests -v
```

The inventory checks positive images against the supplied COCO file, decodes every frame, records SHA-256 hashes, and reports exact duplicates and unreadable files. Derived files go in `outputs/`, which is ignored by Git. `glenda.splitting` builds transitive connected components of known case IDs, video IDs and identical image hashes and checks assignments. It does not create a split or resolve unknown patient identities or near duplicates.

The official [GLENDA v1.5 documentation](https://ftp.itec.aau.at/datasets/GLENDA/) identifies annotated endometriosis frames and sequences with no visible endometriosis-related pathology as binary classification sources. The release is restricted to non-commercial scientific research and requires citation of Leibetseder et al., *GLENDA: Gynecologic Laparoscopy Endometriosis Dataset*, MMM 2020, [doi:10.1007/978-3-030-37734-2_36](https://doi.org/10.1007/978-3-030-37734-2_36).

## Critical limitations

Case IDs are present in positive filenames. Some negative videos match positive videos and inherit their known case ID. The remaining negative videos lack a case mapping, so grouping by available case/video IDs cannot guarantee patient independence. Seek authoritative case metadata before interpreting patient-level generalization. Exact-duplicate checks alone do not rule out near-duplicate leakage. The dataset's curated frame prevalence is not clinical prevalence. Prior reported scores remain unverified, and no test results are claimed here.
