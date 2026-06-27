# Benchmarking Classical Machine-Learning Classifiers for Land-Cover and Surface-Water Mapping on Sentinel-2 Imagery (EuroSAT)

**Author:** Hyrum Hurst
**Affiliation:** Independent researcher
**Contact:** hyrumhurts@gmail.com

> Draft. Numbers written as `{{like_this}}` are filled automatically from
> `data/results.json` (`python src/fill_paper.py`). Every result in this paper
> comes from a single reproducible run of `src/run_study.py`. Read
> `paper/defense_brief.md` before you submit.

---

## Abstract

Mapping land cover and surface water from satellite imagery is a core task in
environmental monitoring, but high-accuracy methods often rely on deep networks
that are computationally heavy and hard to interpret. We ask how far a
transparent, lightweight approach can go: hand-crafted spectral and texture
features fed to classical machine-learning classifiers. Using the EuroSAT
benchmark (`{{n_images}}` Sentinel-2 image patches across `{{n_classes}}`
land-cover classes), we extract `{{n_features}}` interpretable features per image
and train four classifiers, evaluating them on a held-out stratified test set of
`{{n_test}}` images. The best model ({{best_model}}) reached an overall accuracy
of `{{best_accuracy}}`, a macro-averaged F1 of `{{best_macro_f1}}`, and a Cohen's
kappa of `{{best_kappa}}`, against a majority-class baseline of
`{{baseline_accuracy}}`. The two water classes (river and lake/sea) were
classified with a mean F1 of `{{water_f1}}`. The pipeline is fully reproducible,
runs in minutes on a CPU with no specialized hardware or cloud services, and
shows that interpretable features with classical learners remain a strong,
auditable baseline for satellite land-cover classification.

## 1. Introduction

Knowing what is on the ground, and in particular where surface water is, underpins
work on drought, agriculture, and urban growth. In the western United States, for
example, the Colorado River drought has made surface-water monitoring an urgent
question. Satellites such as the European Space Agency's Sentinel-2 image the
entire planet on a regular cadence, and machine learning is the standard tool for
turning that imagery into land-cover maps.

Most state-of-the-art results on land-cover benchmarks come from deep
convolutional neural networks, which are accurate but heavy to train and difficult
to interpret. This study takes the opposite, deliberately simple stance: can a
small set of human-readable features (color statistics, simple spectral ratios,
and texture) combined with classical classifiers reach useful accuracy, while
staying fast and explainable? Such a baseline is valuable precisely because it is
auditable: every feature has a plain meaning, and the model can be inspected.

We evaluate this on EuroSAT (Helber et al., 2019), a widely used benchmark of
Sentinel-2 patches, and we benchmark several classifiers on identical features so
the comparison is fair. We pay particular attention to the water classes, which
connect the study to surface-water monitoring.

## 2. Data

EuroSAT consists of `{{n_images}}` labeled Sentinel-2 image patches (64 by 64
pixels) drawn from across Europe, in `{{n_classes}}` land-cover classes: annual
crop, forest, herbaceous vegetation, highway, industrial buildings, pasture,
permanent crop, residential buildings, river, and sea/lake. We use the RGB version
of the dataset. The class distribution is approximately balanced, so a
majority-class baseline sits near one in ten.

## 3. Methods

### 3.1 Feature extraction

From each image we compute `{{n_features}}` interpretable features:

- per-channel mean and standard deviation (red, green, blue);
- per-channel 10th, 50th, and 90th percentiles;
- three normalized-difference ratios between channels, summarized by mean and
  standard deviation, as lightweight analogues of spectral indices;
- overall brightness mean and standard deviation;
- per-channel 8-bin intensity histograms;
- a texture descriptor from the mean and standard deviation of the brightness
  gradient magnitude.

Every feature has a direct physical or statistical interpretation, which is the
point of the approach.

### 3.2 Classifiers and evaluation

We split the data into a stratified 75/25 training and held-out test set
(`{{n_train}}` train, `{{n_test}}` test), so class proportions are preserved and
the test set is never seen during training. On identical features we train four
classifiers: a random forest, a histogram-based gradient-boosted tree ensemble,
k-nearest neighbors, and multinomial logistic regression. A majority-class dummy
classifier provides the chance baseline. We report overall accuracy, macro
F1 (which weights all classes equally), and Cohen's kappa, plus a per-class F1
breakdown and a confusion matrix for the best model.

### 3.3 Reproducibility

The entire study runs from a single script (`src/run_study.py`) on a CPU, with a
fixed random seed, and downloads the dataset automatically. No GPU, cloud account,
or specialized geospatial service is required.

## 4. Results

The classifiers clearly learn real structure: every model far exceeds the
`{{baseline_accuracy}}` majority-class baseline. The best model,
{{best_model}}, reached an overall accuracy of `{{best_accuracy}}`, a macro F1 of
`{{best_macro_f1}}`, and a kappa of `{{best_kappa}}` on the held-out test set. The
random forest followed at `{{rf_accuracy}}` accuracy (kappa `{{rf_kappa}}`), then
logistic regression at `{{logreg_accuracy}}` and k-nearest neighbors at
`{{knn_accuracy}}`.

The two water classes (river and sea/lake) were recovered with a mean F1 of
`{{water_f1}}`, among the stronger classes, which is encouraging for
surface-water applications: open water has a distinctive, low-variance appearance
that simple features capture well.

**Figure 1.** Held-out confusion matrix for the best model across all
`{{n_classes}}` classes.
**Figure 2.** Per-class F1 for the best model.
**Figure 3.** Held-out accuracy of all classifiers on identical features, with the
majority-class baseline.

## 5. Discussion

A transparent pipeline of interpretable features and classical learners reaches
roughly `{{best_accuracy}}` accuracy on EuroSAT. This is below the high-90s
accuracies reported for deep convolutional networks on the same benchmark, and we
do not claim otherwise. The contribution is a different trade-off: a model that is
fast, runs anywhere, and whose every input is human-readable. For applications
where auditability and low cost matter, or as a sanity-checking baseline before
reaching for a deep model, this is a useful operating point.

The strong water-class performance is notable. Because the study uses only the RGB
bands, it cannot compute a true water index such as NDWI, which relies on
shortwave infrared. That the water classes still separate well suggests that even
visible-band statistics carry a clear water signal, and that adding the infrared
bands (available in the multispectral version of EuroSAT) would likely improve
them further.

## 6. Limitations

- **RGB only.** Without the near- and shortwave-infrared bands, standard water and
  vegetation indices cannot be computed; the multispectral version would likely
  raise accuracy.
- **Hand-crafted features cap accuracy.** Convolutional networks that learn their
  own features outperform this approach on EuroSAT; the gap is the price of
  interpretability and low compute.
- **European imagery.** EuroSAT covers Europe, so the trained model is not
  directly an Arizona or Colorado-River model; it demonstrates the method, not a
  region-specific product.
- **Single split.** Results come from one stratified held-out split with a fixed
  seed; cross-validation would tighten the confidence on the reported numbers.

## 7. Conclusion

Interpretable spectral and texture features with classical classifiers reach
`{{best_accuracy}}` accuracy and a kappa of `{{best_kappa}}` on the EuroSAT
Sentinel-2 land-cover benchmark, with strong water-class performance
(F1 `{{water_f1}}`), all in a fully reproducible CPU-only pipeline. The approach
is a fast, auditable baseline for satellite land-cover and surface-water mapping.

## Data and Code Availability

All code and the analysis script are available at
`https://github.com/hyrumhurst1/sentinel2-landcover-ml`. The EuroSAT dataset is
publicly available via the Hugging Face dataset hub (`{{dataset}}`).

## Author Contributions and AI-Assistance Disclosure

See `paper/ai_disclosure.md`. AI tools assisted with code scaffolding, drafting,
and editing under the author's direction; the author is responsible for the study
design, the interpretation of results, and the accuracy of all claims.

## References

See `paper/references.md`. Verify each citation before submission.
