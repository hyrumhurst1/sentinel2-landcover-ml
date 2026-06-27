# Benchmarking Classical Machine-Learning Classifiers for Land-Cover and Surface-Water Mapping on Sentinel-2 Imagery (EuroSAT)

**Author:** Hyrum Hurst
**Affiliation:** Independent researcher
**Contact:** hyrumhurts@gmail.com

> Draft. Numbers written as `{{like_this}}` are filled automatically from
> `data/results.json` (`python src/fill_paper.py`). Every result comes from a
> single reproducible run of `src/run_study.py`. Read `paper/defense_brief.md`
> before you submit, and rewrite this draft in your own voice.

---

## Abstract

Mapping land cover and surface water from satellite imagery is central to
environmental monitoring, but the highest-accuracy methods rely on deep neural
networks that are computationally heavy and hard to interpret. We ask how far a
transparent, lightweight alternative can go: hand-crafted spectral and texture
features fed to classical machine-learning classifiers. Using the EuroSAT
benchmark (`27000` Sentinel-2 image patches across `10`
land-cover classes), we extract `49` interpretable features per image
and evaluate four classifiers with `5`-fold stratified cross-validation
over the full dataset. The best model (histogram-based gradient boosting) reached a cross-validated
accuracy of `0.8874` (standard deviation `0.0035` across
folds), a macro-averaged F1 of `0.8812`, and a Cohen's kappa of
`0.8747`, against a majority-class baseline of `0.1111`. The
two water classes (river and lake/sea) were classified with a mean F1 of
`0.8901`. The pipeline is fully reproducible, runs in minutes on a CPU with
no specialized hardware or cloud services, and shows that interpretable features
with classical learners remain a strong, auditable baseline for satellite
land-cover classification.

## 1. Introduction

Knowing what is on the ground, and in particular where surface water is, underpins
work on drought, agriculture, and urban growth. In the western United States, the
Colorado River drought has made surface-water monitoring an urgent question.
Satellites such as the European Space Agency's Sentinel-2 image the planet on a
regular cadence, and machine learning is the standard tool for turning that
imagery into land-cover maps.

Most state-of-the-art results on land-cover benchmarks come from deep
convolutional neural networks (CNNs), which are accurate but heavy to train and
hard to interpret. This study takes the opposite, deliberately simple stance: can
a small set of human-readable features (color statistics, simple spectral ratios,
and texture) with classical classifiers reach useful accuracy while staying fast
and explainable? Such a baseline is valuable precisely because it is auditable:
every feature has a plain meaning, and the model can be inspected.

We evaluate this on EuroSAT (Helber et al., 2019), a widely used benchmark of
Sentinel-2 patches, and we benchmark several classifiers on identical features so
the comparison is fair, paying particular attention to the water classes.

## 2. Data

EuroSAT consists of `27000` labeled Sentinel-2 image patches (64 by 64
pixels) drawn from across Europe, in `10` land-cover classes: annual
crop, forest, herbaceous vegetation, highway, industrial buildings, pasture,
permanent crop, residential buildings, river, and sea/lake. We use the RGB version
of the dataset. The class distribution is approximately balanced, so a
majority-class baseline sits near one in ten.

## 3. Methods

### 3.1 Feature extraction

From each image we compute `49` interpretable features: per-channel
mean and standard deviation; per-channel 10th, 50th, and 90th percentiles; three
normalized-difference ratios between channels (lightweight analogues of spectral
indices), summarized by mean and standard deviation; overall brightness mean and
standard deviation; per-channel 8-bin intensity histograms; and a texture
descriptor from the mean and standard deviation of the brightness gradient
magnitude. Every feature has a direct physical or statistical interpretation.

### 3.2 Classifiers and evaluation

On identical features we train four classifiers: a random forest, a
histogram-based gradient-boosted tree ensemble, k-nearest neighbors, and
multinomial logistic regression. A majority-class dummy classifier provides the
chance baseline. We evaluate with `5`-fold stratified cross-validation
over the full dataset, so every image is predicted exactly once while held out,
and we report the mean and standard deviation of accuracy, macro F1, and Cohen's
kappa across folds. For the best model we additionally report a per-class F1
breakdown and a confusion matrix computed from the out-of-fold predictions, and we
rank feature importance from the random forest.

### 3.3 Reproducibility

The entire study runs from a single script (`src/run_study.py`) on a CPU, with a
fixed random seed, and downloads the dataset automatically. No GPU, cloud account,
or specialized geospatial service is required.

## 4. Results

Every classifier far exceeds the `0.1111` majority-class baseline,
confirming that the models learn real structure rather than guessing. Under
`5`-fold cross-validation the best model, histogram-based gradient boosting, reached an
accuracy of `0.8874` (standard deviation `0.0035` across
folds), a macro F1 of `0.8812`, and a kappa of `0.8747`. The
random forest followed at `0.8655` accuracy (kappa `0.8503`), then
logistic regression at `0.8353` and k-nearest neighbors at
`0.8248`.

The two water classes (river and sea/lake) were recovered with a mean F1 of
`0.8901`, among the stronger classes, which is encouraging for
surface-water applications: open water has a distinctive, low-variance appearance
that simple features capture well. The most informative features by random-forest
importance were `texture_grad_mean`, `texture_grad_std`, and `nd_GB_mean`,
which are interpretable color and brightness statistics rather than opaque learned
filters.

**Figure 1.** Cross-validated (out-of-fold) confusion matrix for the best model.
**Figure 2.** Per-class F1 for the best model.
**Figure 3.** Cross-validated accuracy of all classifiers on identical features,
with error bars and the majority baseline.
**Figure 4.** The 15 most informative features by random-forest importance.

## 5. Discussion

A transparent pipeline of interpretable features and classical learners reaches
roughly `0.8874` accuracy on EuroSAT. This is below the 98.57% reported
for the deep convolutional network in the original EuroSAT benchmark (Helber et
al., 2019), and we do not claim otherwise. The contribution is a different
trade-off: a model that is fast, runs anywhere, and whose every input is
human-readable. For applications where auditability and low cost matter, or as a
sanity-checking baseline before reaching for a deep model, this is a useful
operating point.

The strong water-class performance is notable. Because the study uses only the RGB
bands, it cannot compute a true water index such as NDWI (McFeeters, 1996), which
relies on shortwave infrared. That the water classes still separate well suggests
that even visible-band statistics carry a clear water signal, and that adding the
infrared bands (available in the multispectral version of EuroSAT) would likely
improve them further.

## 6. Limitations

- **RGB only.** Without the near- and shortwave-infrared bands, standard water and
  vegetation indices cannot be computed; the multispectral version would likely
  raise accuracy.
- **Hand-crafted features cap accuracy.** Convolutional networks that learn their
  own features outperform this approach on EuroSAT; the gap is the price of
  interpretability and low compute.
- **European imagery.** EuroSAT covers Europe, so the trained model is not directly
  an Arizona or Colorado-River product; it demonstrates the method, not a
  region-specific tool.
- **Single dataset.** Results are on one benchmark; generalization to other sensors
  and regions is not tested here.

## 7. Conclusion

Interpretable spectral and texture features with classical classifiers reach
`0.8874` cross-validated accuracy and a kappa of `0.8747` on the
EuroSAT Sentinel-2 land-cover benchmark, with strong water-class performance
(F1 `0.8901`), all in a fully reproducible CPU-only pipeline. The approach is
a fast, auditable baseline for satellite land-cover and surface-water mapping.

## Data and Code Availability

All code and the analysis script are available at
`https://github.com/hyrumhurst1/sentinel2-landcover-ml`. The EuroSAT dataset is
publicly available via the Hugging Face dataset hub (`blanchon/EuroSAT_RGB`).

## Author Contributions and AI-Assistance Disclosure

See `paper/ai_disclosure.md`. AI tools assisted with code scaffolding, drafting,
and editing under the author's direction; the author is responsible for the study
design, the interpretation of results, and the accuracy of all claims.

## References

See `paper/references.md`. Citations were verified against source.
