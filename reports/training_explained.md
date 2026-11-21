# The longer training experiment, explained

## Why run another experiment?

The first study tried only a small number of settings and its final model trained for one pass through the development data. The later experiment asked a narrower question: **what happens when we allow more training time and adjust different amounts of the same pretrained model?** It used the existing development portion only. It did not evaluate the later models on the already-used internal test or on a new external cohort.

A pretrained model starts with visual patterns learned from a different image dataset. Researchers can then change its internal settings, called *weights*, to suit a new task. This experiment compared three amounts of change:

- **Frozen:** keep the pretrained image-processing part fixed and train a new output layer.
- **Partial:** adjust the output layer and the last part of the image-processing model.
- **Full:** allow the entire model to adjust.

All three used the same model family, learning rate, batch size, weighted loss and development split. Keeping these settings the same makes the comparison easier to interpret, but the chosen learning rate may suit one approach better than another. *Weighted loss* means mistakes on the less common positive frames were given more influence during training; it does not make the resulting scores reliable probabilities.

## What is a fold? What is an epoch?

The development data were divided into five groups called **folds**. For each approach, the researchers trained five models: each model trained on four folds and was checked on the fifth. The reported score for each fold comes from the model that did not train on that fold. Known related image groups were kept together, but unknown patient links remain a limitation.

An **epoch** is one pass through a model's training frames. Each fold could run for up to 30 epochs. If its validation score did not improve for seven epochs in a row, training stopped early. The researchers retained the epoch with the best score on that same validation fold. Since the same fold helps choose the best epoch and supplies the reported score, the reported values are optimistic as estimates of new-data performance.

Across the three approaches and five folds, all **15 runs completed**, using **195 epochs in total**. This is more extensive than the earlier short search; it is not evidence by itself that any model converged or is clinically useful.

## What were the results?

The main comparison uses **average precision (AP)**. AP summarizes the relationship between finding positive frames and avoiding false alarms as the score cutoff changes. It ranges from 0 to 1; higher is generally better for this task. It is not the percentage of positive frames detected at a particular cutoff. See the [metrics guide](metrics_explained.md) for a fuller explanation.

| Approach | Five fold AP scores | Mean AP | What the mean says |
|---|---|---:|---|
| Frozen | 0.515, 0.751, 0.551, 0.411, 0.740 | 0.594 | Average of the five frozen-model fold scores |
| Partial | 0.575, 0.769, 0.767, 0.793, 0.909 | **0.763** | Highest average in this limited comparison |
| Full | 0.727, 0.691, 0.713, 0.582, 0.702 | 0.683 | Between the other two averages |

Partial fine-tuning had the highest mean AP **on these development folds**. The five folds are not five independent external studies, and the numbers vary from fold to fold. The comparison does not prove that partial fine-tuning will perform best on new patients or that the later models improve on the original model's internal test result. The models were compared after the earlier test result was already known, and their scores have not been tested on a fresh independent cohort.

The best epoch differed across folds. For example, the partial models' best epochs were **3, 3, 1, 8 and 1**; they actually ran **10, 10, 8, 15 and 8 epochs**, respectively. The additional epochs include the waiting period for early stopping. In contrast, one frozen model ran all 30 permitted epochs. See the [detailed table](deep_development_results.md) for all approaches and folds.

## What about score reliability?

The report also gives the **Brier score**, the average squared difference between each model score and the 0/1 frame label. Lower is better for this measure. The reported values below are simple averages of the five fold Brier scores, giving each fold the same weight even though the folds contain different numbers of frames.

| Approach | Mean fold Brier score |
|---|---:|
| Frozen | 0.177 |
| Partial | 0.399 |
| Full | 0.415 |

The partial approach's higher AP did not translate into the lowest Brier score. Brier score reflects more than calibration alone, and these development values cannot establish probability reliability on new data. The model scores should not be interpreted as disease probabilities.

The partial models also used very different parts of the score scale: the middle score in one fold was about **0.019**, while in another it was about **0.999**. A cutoff that works for one model may therefore work poorly for another. Those are medians across frames, not estimated disease risks.

## What would be needed next?

First, obtain reliable patient-to-video mappings and more independent sources, especially negative cases. Then specify the intended use and a required operating point, develop and assess calibration using data separate from final evaluation, and test the frozen model and cutoff on a genuinely independent cohort. Do not reuse the earlier internal test to choose or tune the new model. The [next-study protocol](../docs/next_study.md) gives the research plan; the [original study explanation](study_explained.md) describes why these constraints matter.

The [detailed results](deep_development_results.md) and [machine-readable record](deep_development_results.json) contain all fold scores, epoch counts and artifact checksums. The image-level predictions and model weights remain private and are not included in the repository.
