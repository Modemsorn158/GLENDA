# Tuning EfficientNet-B0 for binary classification on GLENDA: interim research report

**Status:** In progress. This report separates a completed dataset audit and feasibility run from planned comparative research. No held-out performance or clinical validation is claimed.

## Abstract

We are rebuilding an image-level binary classification study using GLENDA v1.5. An audit found 373 annotated positive and 13,438 negative frames. Known cases, videos, exact duplicate frames and conservative near-duplicate candidates are linked before grouped splitting. One-epoch frozen-backbone, class-weighting, partial-fine-tuning and ResNet50 feasibility runs completed across five development folds. Longer tuning remains incomplete. Unknown case links in part of the negative set limit claims about patient independence. No test evaluation or external validation has been performed.

## Introduction

The research question is how well transfer learning with EfficientNet-B0 identifies annotated pathology frames versus the release's no-visible-pathology counterexamples, under grouped evaluation and severe frame-level imbalance. This is frame classification, not detection of the location or extent of lesions and not a diagnostic tool.

## Related work

Leibetseder and colleagues introduced GLENDA [1]; the v1.5 release distinguishes annotated keyframes from negative video sequences [2]. Zaidi and colleagues [3] evaluated image classification architectures on GLENDA. Leibetseder and colleagues [4] investigated region-based lesion detection and localization; those tasks and their metrics differ from frame classification. Zhu and Elbattah [5] studied image classifiers and compared visual explanations with lesion masks. Explanation-map overlap is a distinct question from classification accuracy and does not establish clinical validity. Scores from these studies are not directly comparable without aligned data, grouping and evaluation protocols.

## Dataset and audit

The supplied v1.5 files contain 13,811 JPEG frames, including 373 COCO-annotated positive images and 13,438 images from no-pathology sequences. All frame hashes match the corresponding supplied ZIP entries and the COCO JSON matches byte-for-byte ([archive check](archive_check.json)); ancillary archive files were not exhaustively compared. The audit recorded 628 region annotations, 102 positive case identifiers, 205 positive videos and 20 negative videos. Five negative video IDs also occur among positive cases and were linked to those cases. Fifteen negative video IDs have no known case identifier in the supplied files. All frames decoded; 325 byte-identical duplicate sets were found, with no exact cross-label sets. A 64-bit difference-hash screen at Hamming distance at most three identified 15 cross-component candidates; visual inspection showed that some were false matches, so conservative linking reduces available independence without proving image identity. It can miss other near duplicates. The generated inventory and audit summary reside under `outputs/audit/`.

The negative label describes the released counterexample sequences; it is not a histological confirmation for every frame. The 2.7% positive fraction describes this curated release, not an estimate of clinical prevalence. Five negative videos contribute 8,605 of 13,438 negative frames (64.0%), whereas the five largest positive videos contribute 34 of 373 annotated frames (9.1%). The generated `outputs/audit/composition.png` shows this source imbalance, a potential source shortcut and a reason frame-level uncertainty can be misleading.

![Class and source-video composition](../outputs/audit/composition.png)

## Methodology

Groups are transitive components formed from known case and video IDs, exact hashes and candidate links. A fixed seed creates a grouped outer holdout. Initial v1 development folds were used for feasibility studies; after observing that one contained only one negative-bearing group, development groups were reassigned in v2 to give two or three negative-bearing groups per fold. The original holdout was preserved exactly. The first v1 tuning attempt was interrupted after three folds and retained without model selection. Reassignment occurred after exploratory development results and is disclosed here; v1 and v2 folds are not paired. Neither validation nor test frames are undersampled. Known identities do not establish patient independence for the unmapped negative videos. The manifests, source hashes, seeds and distributions are in `outputs/splits/v1/` and `outputs/splits/v2/`.

The model uses a single-logit ImageNet-pretrained EfficientNet-B0 with 224-by-224 resizing, ImageNet normalization and a training-only horizontal flip. One-epoch feasibility comparisons examined frozen versus partial fine-tuning, unweighted versus class-weighted loss, and a frozen ResNet50 baseline. A separate short tuning search is in progress on the revised development folds. Average precision drives validation selection. Threshold selection is defined using development out-of-fold predictions; final fitting uses the median best-epoch count from the selected folds. Both procedures must be completed before any test evaluation.

## Experimental setup

The current environment has an NVIDIA RTX 4050 laptop GPU with 6 GB memory and PyTorch 2.15.0+cu132. The feasibility run used frozen features, AdamW, learning rate 0.0003, batch size 16, unweighted binary cross-entropy and one epoch. Its configuration, weight hashes and fold logs are in `outputs/search/feasibility/`; the configuration is in `configs/feasibility.json`. A longer comparative pilot is defined in `configs/pilot.json` but has not been completed.

## Results

The [generated feasibility table](feasibility_results.md) reports each unweighted development-fold result and descriptive variation directly from recorded outputs. The [matched weighting table](weighting_feasibility.md) reports a one-epoch class-weighting ablation on the same folds, including the negative result on fold 0. A matched [ResNet50 comparison](architecture_feasibility.md) also has fold results in both directions. The [fine-tuning comparison](finetuning_feasibility.md) contrasts frozen and partial features at the same learning rate. These results do not establish convergence, statistical significance, architectural superiority or held-out performance. No final fit, threshold selection, held-out evaluation or external validation has been completed.

## Discussion

The original report describes an undersampled 114-frame test set with a 1:2 class ratio; this does not preserve the release's frame distribution, and 114 is not 10% of 13,811. Its page 10 table evaluates both an earlier image-level-validation script and a later video-grouped-validation script on the same test set, despite a single-evaluation description elsewhere. This test reuse and the absence of executable training scripts in the preserved repository limit interpretation of the reported comparison. Its reported scores and explanations remain historical claims rather than verified results of the rebuild. The new protocol protects known sources and duplicates across folds and preserves the held-out frames, but it cannot eliminate unobserved patient overlap. AUC, calibration and uncertainty are planned for the selected final model; the test set must not guide threshold or model changes.

## Limitations

Images are correlated within videos; patient identities for some negative videos are unavailable. The fixed holdout has only three groups with negative frames, and one group contributes 2,269 of the 2,688 negatives (84.4%). Specificity can therefore be dominated by one source and group-bootstrap intervals may be unstable. Difference hashing is incomplete and can over-link unrelated views. Positive annotated keyframes and negative video sequences are curated differently, inviting source and acquisition shortcuts. The resize changes image aspect ratio. Evaluation on this previously used dataset is not external validation. GPU operations need not be exactly deterministic. Diagnostic utility, calibration in deployment, clinical prevalence and lesion localization remain unestablished.

## Conclusion

The audit and grouped experimental foundation are complete enough to support preliminary feasibility work. The research question remains open until the planned comparisons, preselected threshold, final fit and one-time held-out evaluation are completed and critically assessed.

## References

1. Leibetseder A, Kletz S, Schoeffmann K, Keckstein S, Keckstein J. GLENDA: Gynecologic Laparoscopy Endometriosis Dataset. *MultiMedia Modeling*. 2020; LNCS 11962:439–450. https://doi.org/10.1007/978-3-030-37734-2_36
2. ITEC. GLENDA v1.5 dataset documentation and terms. https://ftp.itec.aau.at/datasets/GLENDA/
3. Zaidi SA, Chouvatut V, Phongnarisorn C, Praserttitipong D. Deep learning based detection of endometriosis lesions in laparoscopic images with 5-fold cross-validation. *Intelligence-Based Medicine*. 2025;11:100230. https://doi.org/10.1016/j.ibmed.2025.100230
4. Leibetseder A, Schoeffmann K, Keckstein J, Keckstein S. Endometriosis detection and localization in laparoscopic gynecology. *Multimedia Tools and Applications*. 2022;81:6191–6215. https://doi.org/10.1007/s11042-021-11730-1
5. Zhu Y, Elbattah M. Explainable Deep Learning for Endometriosis Classification in Laparoscopic Images. *BioMedInformatics*. 2025;5(4):63. https://doi.org/10.3390/biomedinformatics5040063
