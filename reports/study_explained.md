# The original GLENDA study, explained

## The question

Can a computer model learn to tell apart frames annotated for endometriosis and frames from videos released as showing no visible endometriosis-related pathology? The project treats this as a **frame classification** task: given one still image, the model produces a score and compares it with a cutoff. It does not draw a boundary around a lesion, review a complete operation or diagnose a person.

**The answer so far is not good enough for clinical use.** In the one-time internal test, the model missed 55 of 75 annotated positive frames. Its scores were poorly calibrated, and the study does not establish how it would perform on new patients or at another hospital.

## What images were used?

The supplied GLENDA v1.5 dataset contained 13,811 frames. Of these, 373 were positive frames with annotations and 13,438 came from videos released as showing no visible endometriosis-related pathology. The latter is a description of these video sequences, not proof that every person was free of disease. The release is curated for research: its share of positive frames is not an estimate of how often endometriosis occurs in patients.

The researchers checked image files against the supplied archives, looking for missing or changed frames. The check matched all 13,811 frames and the annotation file; it did not exhaustively verify every other archive file. They also looked for identical and visually similar images. Similarity checks can miss related images or mistakenly link unrelated ones.

The images were divided into two parts:

| Portion | Positive frames | Negative frames | Purpose |
|---|---:|---:|---|
| Development | 298 | 10,750 | Train models and choose settings |
| Internal test | 75 | 2,688 | Evaluate the chosen model once |

These are **frame counts, not patient counts**. Frames from the same video or case can look alike. The split grouped known related images together, but some negative videos have no known patient identifier. Patient overlap between portions therefore remains possible. The test is *internal* because it comes from the same previously studied dataset, not a new hospital or independent collection.

## How did the first study work?

1. The researchers started with a model that had learned general image features from another dataset and adapted it to these frames.
2. They used development data to compare settings, decide when to stop training and choose the score cutoff. A cutoff is the point above which a score is called positive.
3. They trained the chosen model on all development frames and evaluated it once on the reserved internal test images. The cutoff was fixed before that evaluation.

The cutoff was about **0.99997 on a 0-to-1 score scale**. That number is a decision rule, not a 99.997% chance of disease: the model's scores are not calibrated probabilities. The choice was made from scores produced by several development models, so the same cutoff may behave differently when applied to the final model.

## What happened on the internal test?

| Actual frame label | Model flag | What this means | Frames |
|---|---|---|---:|
| Positive | Positive | Correctly flagged positive | 20 |
| Positive | Negative | **Missed positive** | **55** |
| Negative | Positive | False alarm | 0 |
| Negative | Negative | Correctly not flagged | 2,688 |

The model flagged **20 of the 75 positive frames**. This is a *sensitivity* of 20 ÷ 75 = **26.7%**. It missed the other 73.3%. It also flagged **0 of 2,688 negative frames**. Those observations describe the frames in this test; they do not establish how often a different patient or video would be missed or falsely flagged.

Its accuracy was 2,708 correct frames out of 2,763, or **98.0%**. Since 2,688 of the frames were negative, an always-negative model would score **97.3% accuracy** and still miss every positive. The high accuracy therefore hides the poor positive detection at this cutoff.

The model's **average precision was 0.657** and **ROC-AUC was 0.951**. Both describe how well its scores rank positive frames above negative frames across cutoffs, in different ways. They do not mean that sensitivity was 65.7% or 95.1%, and they do not override the observed 26.7% sensitivity at the chosen cutoff. See the [metrics guide](metrics_explained.md) for definitions and examples.

The **Brier score was 0.479**, where a smaller value means scores are closer to the 0/1 labels on average. A simple constant score based on the development set's positive-frame fraction had a Brier score of **0.026** on the same test. The contrast is a warning about treating the model's scores as probabilities. Brier score combines several aspects of prediction quality, so the contrast alone does not isolate the cause.

## How certain are the numbers?

The test contained **22 known groups of linked images**, but only **three contained negative frames**. One of those three supplied 2,269 of the 2,688 negative frames: **84.4%**. A model that handles these few sources well may behave differently on other videos, patients or equipment.

The [technical results](test_results.md) include *bootstrap intervals*: ranges produced by repeatedly resampling the available known groups and recalculating each metric. For sensitivity the reported range was about **13% to 40%**. It reflects variability under that resampling method; it is not a guarantee that the true value is in that range, especially because the groups are few and some patient relationships are unknown. Some resamples do not contain the labels needed for a metric, so the table also shows the number of valid repeats. For example, specificity was defined in 960 of 1,000 repeats.

The reported intervals for precision and specificity both collapse to 100%–100% because the observed test had no false alarms to resample. That is a limitation of this particular bootstrap result, not evidence that future performance will be perfect.

## What happened next?

A later development-only experiment trained models for longer and compared different amounts of fine-tuning. It did not reevaluate the consumed test set. Read [the training experiment](training_explained.md) for its results and why they cannot prove an improvement on new data.

Before stronger claims, the project needs verified patient-to-video mappings, more independent negative sources, an agreed purpose and operating requirement, and an independent external cohort. A separate dataset is also needed to develop and assess any calibration approach. The detailed [next-study protocol](../docs/next_study.md) describes these requirements.

## Sources and further reading

The dataset was introduced by Leibetseder and colleagues, *GLENDA: Gynecologic Laparoscopy Endometriosis Dataset*, MMM 2020, [doi:10.1007/978-3-030-37734-2_36](https://doi.org/10.1007/978-3-030-37734-2_36). The [dataset release page](https://ftp.itec.aau.at/datasets/GLENDA/) states its research-use and attribution terms.

For the detailed study, see the [research report](research_report.md), [test results](test_results.md), [error analysis](error_analysis.md) and [split composition](split_v2.md). The [metrics guide](metrics_explained.md) explains every metric in the main test table.
