"""
run_study.py
------------
A self-contained satellite-image machine-learning study, NO cloud/Earth Engine.

Task: classify Sentinel-2 land cover (10 classes, including surface water) on the
EuroSAT benchmark using hand-crafted spectral/texture features + classical ML
classifiers, with a held-out test set and honest metrics.

Outputs: data/results.json, figures/*.png. Runs end to end on CPU in minutes.
"""
import json
import os
import time

import numpy as np
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# 1. Load EuroSAT RGB (real Sentinel-2 imagery), robust to source/split layout
# --------------------------------------------------------------------------
def load_eurosat():
    last = None
    for repo in ("blanchon/EuroSAT_RGB", "giswqs/EuroSAT_RGB", "timm/eurosat-rgb"):
        try:
            log(f"Loading dataset {repo} ...")
            ds = load_dataset(repo)
            parts = [ds[s] for s in ds.keys()]
            full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            log(f"Loaded {repo}: {len(full)} images, splits={list(ds.keys())}")
            return full, repo
        except Exception as e:  # noqa
            last = e
            log(f"  {repo} failed: {type(e).__name__}: {e}")
    raise SystemExit(f"All dataset sources failed. Last error: {last}")


full, source = load_eurosat()

# Detect image + label columns.
cols = full.column_names
img_col = next((c for c in ("image", "img", "rgb", "jpg") if c in cols), None)
lbl_col = next((c for c in ("label", "labels", "class") if c in cols), None)
if img_col is None or lbl_col is None:
    raise SystemExit(f"Could not find image/label columns in {cols}")
try:
    class_names = full.features[lbl_col].names
except Exception:
    class_names = sorted({str(x) for x in full[lbl_col]})
log(f"image col='{img_col}', label col='{lbl_col}', classes={class_names}")


# --------------------------------------------------------------------------
# 2. Feature extraction (spectral statistics + simple texture per image)
# --------------------------------------------------------------------------
def features_from_image(arr):
    """arr: HxWx3 uint8 -> 1D feature vector (spectral + texture)."""
    a = arr.astype(np.float32) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    eps = 1e-6
    feats = []
    # per-channel mean/std
    for ch in (R, G, B):
        feats += [ch.mean(), ch.std()]
    # per-channel percentiles 10/50/90
    for ch in (R, G, B):
        feats += list(np.percentile(ch, [10, 50, 90]))
    # pseudo spectral indices (normalized differences) mean + std
    for num, den in ((G - R, G + R), (G - B, G + B), (R - B, R + B)):
        nd = num / (den + eps)
        feats += [nd.mean(), nd.std()]
    # brightness
    bright = a.mean(axis=2)
    feats += [bright.mean(), bright.std()]
    # per-channel 8-bin histograms
    for ch in (R, G, B):
        h, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
        feats += list(h / (h.sum() + eps))
    # texture: gradient magnitude of brightness
    gy, gx = np.gradient(bright)
    gm = np.sqrt(gx * gx + gy * gy)
    feats += [gm.mean(), gm.std()]
    return np.asarray(feats, dtype=np.float32)


log("Extracting features ...")
t0 = time.time()
X, y = [], []
n = len(full)
for i in range(n):
    row = full[i]
    arr = np.asarray(row[img_col].convert("RGB"))
    X.append(features_from_image(arr))
    y.append(int(row[lbl_col]))
    if (i + 1) % 3000 == 0:
        log(f"  {i + 1}/{n} images ({(i + 1) / (time.time() - t0):.0f}/s)")
X = np.vstack(X)
y = np.asarray(y)
log(f"Feature matrix: {X.shape}, {X.shape[1]} features/image, "
    f"{time.time() - t0:.0f}s")

# --------------------------------------------------------------------------
# 3. Held-out stratified split (the defensible part)
# --------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)
log(f"Train: {X_tr.shape[0]} | Test: {X_te.shape[0]} (stratified 75/25)")

# --------------------------------------------------------------------------
# 4. Train + compare classifiers on identical features
# --------------------------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=SEED),
    "HistGradBoost": HistGradientBoostingClassifier(random_state=SEED),
    "kNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "LogReg": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, n_jobs=-1)),
    "Baseline(majority)": DummyClassifier(strategy="most_frequent"),
}

results = {}
for name, clf in models.items():
    log(f"Training {name} ...")
    tt = time.time()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    results[name] = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_te, pred)),
        "train_seconds": round(time.time() - tt, 1),
    }
    log(f"  {name}: acc={results[name]['accuracy']:.4f} "
        f"macroF1={results[name]['macro_f1']:.4f} "
        f"kappa={results[name]['kappa']:.4f}")

best_name = max((k for k in results if not k.startswith("Baseline")),
                key=lambda k: results[k]["accuracy"])
log(f"Best model: {best_name}")

# Refit best for detailed report + confusion matrix + importances.
best = models[best_name]
pred = best.predict(X_te)
report = classification_report(y_te, pred, target_names=class_names,
                               output_dict=True, zero_division=0)
cm = confusion_matrix(y_te, pred)

# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (1) Confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (true label)")
ax.set_title(f"{best_name} held-out confusion matrix\n"
             f"acc={results[best_name]['accuracy']:.3f} "
             f"kappa={results[best_name]['kappa']:.3f}")
thresh = cm.max() / 2
for r in range(cm.shape[0]):
    for c in range(cm.shape[1]):
        ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=7,
                color="white" if cm[r, c] > thresh else "black")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_confusion.png"), dpi=160)

# (2) Per-class F1
f1s = [report[c]["f1-score"] for c in class_names]
fig, ax = plt.subplots(figsize=(9, 4))
order = np.argsort(f1s)
ax.barh([class_names[i] for i in order], [f1s[i] for i in order], color="teal")
ax.set_xlabel("F1 score"); ax.set_xlim(0, 1)
ax.set_title(f"{best_name} per-class F1 (held-out test)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_per_class_f1.png"), dpi=160)

# (3) Model comparison
fig, ax = plt.subplots(figsize=(8, 4))
names = list(results.keys())
accs = [results[k]["accuracy"] for k in names]
ax.bar(range(len(names)), accs, color="#4ea1ff")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Held-out accuracy"); ax.set_ylim(0, 1)
ax.set_title("Classifier comparison on EuroSAT (identical features)")
for i, a in enumerate(accs):
    ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_model_comparison.png"), dpi=160)

# (4) RF feature importances (if available)
if hasattr(best, "feature_importances_"):
    imp = best.feature_importances_
    top = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f"f{j}" for j in top], imp[top], color="darkorange")
    ax.set_title(f"{best_name} top-15 feature importances")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_feature_importance.png"), dpi=160)

# --------------------------------------------------------------------------
# 6. Save results.json (+ flat paper_tokens for the writeup)
# --------------------------------------------------------------------------
water_classes = [c for c in class_names
                 if any(w in c.lower() for w in ("water", "river", "lake", "sea"))]
water_f1 = float(np.mean([report[c]["f1-score"] for c in water_classes])) \
    if water_classes else None

out = {
    "dataset": source,
    "n_images": int(n),
    "n_features": int(X.shape[1]),
    "n_classes": len(class_names),
    "classes": list(class_names),
    "split": "stratified 75/25 held-out",
    "n_train": int(X_tr.shape[0]),
    "n_test": int(X_te.shape[0]),
    "models": results,
    "best_model": best_name,
    "water_classes": water_classes,
    "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in class_names},
    "confusion_matrix": cm.tolist(),
    "paper_tokens": {
        "dataset": source,
        "n_images": int(n),
        "n_features": int(X.shape[1]),
        "n_classes": len(class_names),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "best_model": best_name,
        "best_accuracy": round(results[best_name]["accuracy"], 4),
        "best_macro_f1": round(results[best_name]["macro_f1"], 4),
        "best_kappa": round(results[best_name]["kappa"], 4),
        "rf_accuracy": round(results["RandomForest"]["accuracy"], 4),
        "rf_kappa": round(results["RandomForest"]["kappa"], 4),
        "logreg_accuracy": round(results["LogReg"]["accuracy"], 4),
        "knn_accuracy": round(results["kNN(k=5)"]["accuracy"], 4),
        "baseline_accuracy": round(results["Baseline(majority)"]["accuracy"], 4),
        "water_f1": round(water_f1, 4) if water_f1 is not None else None,
    },
}
with open(os.path.join(DATA, "results.json"), "w") as f:
    json.dump(out, f, indent=2)

log("=== DONE ===")
print(json.dumps(out["paper_tokens"], indent=2))
