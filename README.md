# GLENDA: exploring AI for endometriosis images

This project investigates whether an image-classification model can distinguish two kinds of laparoscopic images in the GLENDA research dataset: frames annotated for endometriosis and frames from videos released as showing no visible endometriosis-related pathology. A *frame* is one still image from a video. The model gives each frame a score; it does not locate lesions or make a diagnosis.

[Endometriosis](https://www.who.int/news-room/fact-sheets/detail/endometriosis) is a condition in which tissue similar to the lining of the uterus grows outside it. [Laparoscopy](https://www.nhs.uk/tests-and-treatments/laparoscopy/) is a type of surgery that uses a camera through a small cut to look inside the abdomen or pelvis. GLENDA contains still images from this type of video.

**This is a research project, not a medical device. It must not be used to diagnose patients or guide treatment.** In the completed test, the model missed most of the annotated positive frames. Its scores are not reliable probabilities, and its performance has not been confirmed on an independent external dataset.

## What did the project find?

The first study set aside a group of images for a one-time internal test. At the decision threshold chosen before that test, the results were:

| In the test images | Model result | Count |
|---|---|---:|
| Annotated positive | Flagged as positive | 20 |
| Annotated positive | Missed | **55** |
| Released as negative | Flagged as positive | 0 |
| Released as negative | Not flagged | 2,688 |

The model found **20 of 75 positive frames**, or **26.7%**. The 0 false positives are not proof that the model will avoid false alarms elsewhere: the 2,688 negative frames came from only three known groups of related images, and one group supplied most of them. These are counts of images, not patients.

The model was right on 98.0% of the test frames overall, but most frames were negative. An imaginary model that always says “negative” would also be right on 97.3% of these frames while finding none of the positives. This is why accuracy alone is a poor summary here.

A later experiment trained models for longer and compared three ways of adjusting a pretrained model. The strongest average result in that **development** comparison came from adjusting part of the model. Those figures are exploratory: the later models were not evaluated on a new, independent test set, and their development scores cannot be compared directly with the original test score as proof of improvement.

## Where to start

| If you want to… | Read |
|---|---|
| Understand the dataset, the original test and its numbers | [The study, explained](reports/study_explained.md) |
| Understand the longer training comparison | [The training experiment, explained](reports/training_explained.md) |
| Look up terms such as sensitivity, F1 and Brier score | [A plain-language guide to the metrics](reports/metrics_explained.md) |
| See the detailed research methods and recorded results | [Research report](reports/research_report.md), [original test results](reports/test_results.md), and [extended development results](reports/deep_development_results.md) |
| Install the code, run tests or understand the safeguards | [Technical guide](docs/technical-guide.md) |
| Understand what a responsible next study would need | [Next-study protocol](docs/next_study.md) |
| See how the latest audit was addressed and what remains | [Audit response](reports/comprehensive_audit_response.md) |

## What is in the dataset?

The audited GLENDA v1.5 release contained **13,811 frames**: **373 annotated positive** frames and **13,438 negative** frames. The negative label describes the source videos as having no *visible* endometriosis-related pathology; it is not a guarantee about every patient's health. These curated images are not a representative sample of all patients or clinical settings.

Many frames come from the same video or known case, so they are not as independent as 13,811 separate patient observations. The project grouped images using the information available, but some negative videos have no known patient mapping. It therefore cannot rule out patient overlap across the development and test portions.

## Can I try the project?

The current checkout contains research code, reports and tests, but not the dataset or trained checkpoints needed to run the demonstration. **Older commits in the repository's Git history do contain dataset images.** Removing images from the current checkout does not remove them from earlier commits; access to that history depends on the repository host's permissions and the dataset's terms. The [technical guide](docs/technical-guide.md) explains how to install dependencies, audit authorized data, run tests and set up a new study. The original test has already been used: do not reuse it to select a model, adjust a threshold or claim a fresh evaluation.

The project’s own software is available under the [MIT License](LICENSE). The [GLENDA dataset](https://ftp.itec.aau.at/datasets/GLENDA/) has separate non-commercial scientific research and attribution requirements, and third-party pretrained weights have their own terms. The [study report](reports/study_explained.md) provides the dataset citation.
