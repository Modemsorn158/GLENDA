# Holdout score and error analysis

This is descriptive analysis of the saved one-time evaluation; no threshold or model was changed.

The model Brier score was **0.4793**. A constant score equal to the development positive-frame fraction (0.0270) has Brier score **0.0264** on the same holdout. This is a calibration reference, not a clinical comparator.

The median model score was 0.7085; its 25th and 75th percentiles were 0.3607 and 0.9524. The model scores are not calibrated probabilities.

The 55 false-negative frames came from 20 known linked groups; the largest such group contributed 7 frames. These are grouped frame errors, not confirmed independent patients.

No false positives were observed; this does not prove perfect specificity. Only 3 known groups supplied negative frames, and the threshold was selected from development-fold score scales; it may not transfer reliably to a refitted model.
