# Sentinel-2 Land-Cover & Surface-Water Classification (EuroSAT)

A reproducible, CPU-only machine-learning study: classify land cover and surface
water in 27,000 real Sentinel-2 images (the EuroSAT benchmark) using interpretable
features and classical classifiers, with honest held-out evaluation.

**Author:** Hyrum Hurst · Independent researcher

## Headline result
Best model (histogram gradient boosting): **88.4% accuracy, kappa 0.871, macro-F1
0.878** on a held-out test set of 6,750 images, versus an 11% majority baseline.
Water classes (river + sea/lake): **F1 0.879**. Full numbers in
[`data/results.json`](data/results.json).

## Reproduce it (a few minutes, CPU only, no GPU/cloud)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe datasets scikit-learn pillow numpy matplotlib scipy
.venv/Scripts/python.exe src/run_study.py     # downloads EuroSAT, trains, evaluates, writes outputs
.venv/Scripts/python.exe src/fill_paper.py    # fills paper/paper.md with the real numbers
```
(Or use any Python 3.11+ with `pip install -r requirements.txt`.)

## Layout
```
src/run_study.py     the whole study: load -> features -> train -> evaluate -> save
src/fill_paper.py    fills paper/paper.md {{tokens}} from results.json
paper/paper.md       the manuscript draft
paper/references.md  citations (verify before submitting)
paper/ai_disclosure.md
data/results.json    full machine-readable results
figures/*.png        confusion matrix, per-class F1, model comparison
```

## Method in one line
49 interpretable features per image (color statistics, channel ratios, texture) ->
random forest / gradient boosting / kNN / logistic regression -> held-out metrics.

## License
MIT (code). Text and figures: the author's, pending publication. EuroSAT data is
publicly available under its own terms.
