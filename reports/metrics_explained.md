# A plain-language guide to the numbers

These definitions use the original one-time internal test: 75 annotated positive frames and 2,688 frames from negative video sequences. At the chosen cutoff, the model flagged 20 positive and 0 negative frames; it missed 55 positive frames and did not flag 2,688 negative frames. A *frame* is a still image, not a patient. The more detailed [test table](test_results.md) contains the recorded values and uncertainty intervals.

## First, the four possible outcomes

| Term | Meaning | Original test |
|---|---|---:|
| True positive (TP) | A positive frame was flagged | 20 |
| False negative (FN) | A positive frame was missed | 55 |
| False positive (FP) | A negative frame was flagged | 0 |
| True negative (TN) | A negative frame was not flagged | 2,688 |

“Positive” and “negative” refer to the labels in this dataset. A negative frame comes from a sequence released as showing no visible endometriosis-related pathology; it is not a medical diagnosis.

## Measures at one chosen cutoff

A **score cutoff** or **threshold** converts scores into flags. A score at or above the cutoff is called positive. Changing the cutoff can change every measure in this section. The original study fixed its cutoff before the internal test; it was approximately 0.99997. The score is not a calibrated probability.

| Measure | Plain-language question | Original result and calculation |
|---|---|---|
| **Accuracy** | Of all frames, how many were classified correctly? | 98.0% = (20 + 2,688) ÷ 2,763. It looks high because most frames were negative. |
| **Sensitivity** or **recall** | Of the positive frames, how many were flagged? | 26.7% = 20 ÷ (20 + 55). The model missed the rest. |
| **Specificity** | Of the negative frames, how many were not flagged? | 100% = 2,688 ÷ (2,688 + 0) in this test. This does not guarantee future specificity. |
| **Precision** | Of the frames the model flagged, how many were positive? | 100% = 20 ÷ (20 + 0) in this test. If no frames were flagged, precision would be undefined. |
| **F1** | How does one number balance precision and recall? | 0.421. It is the harmonic mean of precision and recall: `2 × precision × recall ÷ (precision + recall)`. A high value requires both to be high. |
| **Balanced accuracy** | What is the average of sensitivity and specificity, giving both labels equal weight? | 63.3% = (26.7% + 100%) ÷ 2. |

These measures answer different questions. For instance, precision looks only at flagged frames, whereas sensitivity looks at all positive frames. With no false alarms in this particular test, precision was 100% even though the model missed most positives. Precision also depends on the mix of positive and negative examples, which may differ in another setting.

The test report also shows results at a **0.5 reference cutoff**: it flagged all 75 positive frames but also flagged 1,702 negative frames, so precision was about 4.2%. This is a descriptive comparison using the already-consumed test, not a recommendation to change the cutoff or evidence that 0.5 would work on new data.

## Measures across many cutoffs

Models can assign useful relative rankings even when one particular cutoff works poorly. These measures look at rankings across cutoffs, rather than describing the single decision rule above.

- **ROC-AUC (0.951 in the original test):** A summary of how sensitivity and the false-alarm rate change as the cutoff moves. It can also be interpreted as the probability that a randomly chosen positive frame receives a higher score than a randomly chosen negative frame, with ties counted halfway. It is not the chance that a patient has disease, and it is not the sensitivity at the chosen cutoff.
- **Average precision, or AP (0.657 in the original test):** A summary of precision as recall increases while the cutoff changes. It is useful for examining rankings when positive frames are uncommon, but it depends on the positive/negative mix in the evaluated data. It is not the precision at the chosen cutoff or a percentage of patients detected.

Neither ranking measure by itself tells us that a usable decision cutoff exists for a new population. The original model had a high ROC-AUC but missed 55 of 75 positive frames at its selected cutoff.

## Scores as probabilities: calibration and Brier score

A score is **calibrated** as a probability if, among many examples given a particular probability, the corresponding fraction are actually positive. For example, among many images given a calibrated score of 0.2, about 20% would be positive in the relevant population. This is an illustration, not an established property of this model. Calibration can change with the population and how examples are selected.

The **Brier score** is the average of `(score − label)²`, with label 1 for a positive frame and 0 for a negative frame. Lower is better: a confident score near 1 for a negative frame incurs a large penalty. The original test score was **0.479**. A constant score equal to the development set's positive-frame fraction had a Brier score of **0.026** on the same test. That constant is a simple reference, not a competing clinical model. Brier score combines calibration and other aspects of prediction quality; it does not isolate calibration by itself.

The longer training report averages five fold-level Brier scores without weighting by the number of frames in each fold. That average describes those fold models on their development validation frames; it is not the Brier score of one final model on an independent cohort.

## How to read ranges and variation

- A **fold** is one part of the development data held aside while a model trains on the other parts. The five scores in the longer training report come from five separately trained models. Their mean is the arithmetic average. The **sample standard deviation** (SD) describes how spread out those five scores are; it is not a confidence interval, and five folds are not five external studies.
- An **epoch** is one pass over the training portion for a fold. A **best epoch** is the epoch with the highest validation AP for that fold. **Epochs run** includes later epochs used to decide whether to stop after a period without improvement.
- A **median score** is the middle score after ordering a fold's frame scores. It describes the score distribution, not the probability of disease. Large differences in medians between fold models warn that one shared cutoff may not transfer well.
- A **bootstrap interval** is computed by repeatedly resampling the available groups and recalculating a metric. The reported 95% interval is the range between selected lower and upper resampled values under that procedure. It is conditional on the available data and assumptions; it is not a guarantee about performance elsewhere.
- **Valid replicates** count resamples for which a metric could be calculated. A resample without negative frames, for example, cannot produce specificity. In the original table, specificity was defined in 960 of 1,000 resamples.

There were only three known negative-containing groups in the original test. Because none produced a false alarm, resampling those same observations yields a 100%–100% specificity interval. That interval does not establish perfect specificity in new videos or patients. Unknown patient links and repeated frames further limit the meaning of these ranges.
