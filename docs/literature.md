# Literature notes

These notes distinguish tasks and avoid comparing numerical scores across different cohorts or split protocols.

## Dataset and labels

Leibetseder et al. [1] introduced GLENDA for computer-aided analysis of gynecologic laparoscopic images. The v1.5 documentation [2] describes annotated endometriosis keyframes and no-visible-pathology video sequences, and lists both binary classification and localization as possible uses. The release is research-only and requires attribution. A negative frame label is a curated visual label, not an assertion of histological absence.

## Image classification

Zaidi et al. [3] describe VGG19, ResNet50 and InceptionV3 for pathological versus non-pathological frame classification with five-fold cross-validation. This is closer to the present task than lesion localization. Its reported metrics do not by themselves establish performance on our held-out groups; dataset selection, sampling, source grouping, augmentation and prevalence must be aligned before numerical comparisons are meaningful.

Zhu and Elbattah [4] describe several classification architectures, including ResNet50 and EfficientNet-B2, alongside Grad-CAM and SHAP analyses compared with lesion masks. Their work motivates examining model attention, but a visual explanation is not itself a lesion detector or proof of causal reasoning. The current rebuild does not claim an explanation-map study.

## Detection and localization

Leibetseder et al. [5] investigated region-based detection and segmentation with Faster R-CNN and Mask R-CNN. Their localization metrics evaluate spatial predictions, whereas a binary frame classifier supplies no bounding box or segmentation mask. Neither task's reported score should be substituted for the other.

## Relevance to this protocol

The literature motivates transfer learning and careful source-level separation, but it does not resolve the missing patient mapping in the supplied negative sequences. Precision–recall analysis can reveal consequences of class imbalance that a ROC curve alone may obscure [6]; average precision remains dependent on the evaluated prevalence. Cluster bootstrap methods are designed for correlated observations [7], but their usefulness depends on appropriate, sufficiently numerous independent clusters. In this study the test set has only three known negative-bearing groups and unknown patient links, so bootstrap intervals must be interpreted cautiously. This study therefore reports known-group separation, frame and group counts, class imbalance and uncertainty without asserting patient-independent or external validation. Its one-epoch feasibility results are exploratory and not a superiority comparison with published studies.

## References

1. Leibetseder A, Kletz S, Schoeffmann K, Keckstein S, Keckstein J. GLENDA: Gynecologic Laparoscopy Endometriosis Dataset. *MultiMedia Modeling*. 2020; LNCS 11962:439–450. https://doi.org/10.1007/978-3-030-37734-2_36
2. ITEC. GLENDA v1.5 dataset documentation. https://ftp.itec.aau.at/datasets/GLENDA/
3. Zaidi SA, Chouvatut V, Phongnarisorn C, Praserttitipong D. Deep learning based detection of endometriosis lesions in laparoscopic images with 5-fold cross-validation. *Intelligence-Based Medicine*. 2025;11:100230. https://doi.org/10.1016/j.ibmed.2025.100230
4. Zhu Y, Elbattah M. Explainable Deep Learning for Endometriosis Classification in Laparoscopic Images. *BioMedInformatics*. 2025;5(4):63. https://doi.org/10.3390/biomedinformatics5040063
5. Leibetseder A, Schoeffmann K, Keckstein J, Keckstein S. Endometriosis detection and localization in laparoscopic gynecology. *Multimedia Tools and Applications*. 2022;81:6191–6215. https://doi.org/10.1007/s11042-021-11730-1
6. Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLoS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
7. Rutter CM. Bootstrap estimation of diagnostic accuracy with patient-clustered data. *Academic Radiology*. 2000;7(6):413–419. https://doi.org/10.1016/S1076-6332(00)80381-5
