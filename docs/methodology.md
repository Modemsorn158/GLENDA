# Protocol and open limitations

## Intended evaluation

Before selecting models, freeze one grouped holdout and use only the development folds for model selection, early stopping and threshold selection. The current split generator uses the first shuffled five-way `StratifiedGroupKFold` partition as the holdout and a separate five-way partition for development. The seed and source file hashes are recorded. No images are subsampled. Report the resulting distribution rather than describing it as clinical prevalence; a grouped split can differ substantially from the overall frame distribution. Groups are not independent patients where negative case identifiers are unavailable.

Link all frames with the same known case, video, exact SHA-256 digest, or difference-hash screening candidate before splitting. The dHash threshold of three is a reproducible, conservative screening rule, not a diagnosis of duplication; a visual review found some candidate pairs are plainly different. It does not detect all transformed or near-identical images. It can merge independent groups, and unknown patient relationships remain a leakage threat. Seek the dataset owners' case mapping and use stronger duplicate review before claiming patient independence.

## Historical report

The supplied 15-page report describes a 114-frame test set with 38 positive and 76 negative images, and a 1:2 sampling ratio. That is not 10% of the 13,811 released frames. It reports grouped folds and a later grouped internal validation correction, but its stated scores are historical claims, not verified outputs of this rebuild. It also acknowledges the altered test prevalence and lack of ROC/PR curves. Its optimizer and learning-rate interpretations should not be treated as causal without controlled comparisons. Thai text extraction is imperfect; interpretation should be checked against the original pages.

## Status

Two CPU smoke runs each used eight development frames and one epoch with frozen EfficientNet-B0 features: one randomly initialized and one initialized with ImageNet weights. These verify execution only. One dataset-scale, one-epoch frozen ImageNet EfficientNet-B0 feasibility run was completed on development fold 0, using the RTX 4050 and unweighted loss. Its validation average precision was 0.3938 on 2,150 validation frames (60 positive); this is a preliminary single-fold observation, not a model comparison or test result. Configuration and logs are in `outputs/runs/pilot_fold0_1epoch/`. No hyperparameter search, final model or test evaluation has been completed. A split manifest alone does not make the evaluation independent, clinically representative, or prospectively locked. Model and threshold choices must be fixed without using test predictions. External data are required for an external validation claim.

Dataset: Leibetseder et al., *GLENDA: Gynecologic Laparoscopy Endometriosis Dataset*, MMM 2020, https://doi.org/10.1007/978-3-030-37734-2_36. Dataset documentation: https://ftp.itec.aau.at/datasets/GLENDA/ (non-commercial scientific research only; citation required).
