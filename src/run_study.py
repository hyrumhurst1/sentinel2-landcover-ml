"""
run_study.py
------------
Self-contained satellite-image ML study (NO cloud / Earth Engine).

Task: classify Sentinel-2 land cover and surface water on EuroSAT using
interpretable spectral/texture features + classical classifiers, evaluated with
5-fold stratified cross-validation (robust) and out-of-fold predictions.

Outputs: data/results.json, data/features.npz (cache), figures/*.png.
Runs on CPU in minutes.
"""
import json
import os
import time

import numpy as np
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (make_scorer, cohen_kappa_score, f1_score,
                             accuracy_score, confusion_matrix, classification_report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
SEED = 42
CACHE = os.path.join(DATA, "features.npz")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# Feature names (must match features_from_image order). 49 interpretable feats.
# --------------------------------------------------------------------------
FEATURE_NAMES = (
    ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std"]
    + ["R_p10", "R_p50", "R_p90", "G_p10", "G_p50", "G_p90",
       "B_p10", "B_p50", "B_p90"]
    + ["nd_GR_mean", "nd_GR_std", "nd_GB_mean", "nd_GB_std",
       "nd_RB_mean", "nd_RB_std"]
    + ["bright_mean", "bright_std"]
    + [f"R_hist{i}" for i in range(8)]
    + [f"G_hist{i}" for i in range(8)]
    + [f"B_hist{i}" for i in range(8)]
    + ["texture_grad_mean", "texture_grad_std"]
)


def features_from_image(arr):
    a = arr.astype(np.float32) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    eps = 1e-6
    feats = []
    for ch in (R, G, B):
        feats += [ch.mean(), ch.std()]
    for ch in (R, G, B):
        feats += list(np.percentile(ch, [10, 50, 90]))
    for num, den in ((G - R, G + R), (G - B, G + B), (R - B, R + B)):
        nd = num / (den + eps)
        feats += [nd.mean(), nd.std()]
    bright = a.mean(axis=2)
    feats += [bright.mean(), bright.std()]
    for ch in (R, G, B):
        h, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
        feats += list(h / (h.sum() + eps))
    gy, gx = np.gradient(bright)
    gm = np.sqrt(gx * gx + gy * gy)
    feats += [gm.mean(), gm.std()]
    return np.asarray(feats, dtype=np.float32)


# --------------------------------------------------------------------------
# 1. Load EuroSAT + extract features (cached)
# --------------------------------------------------------------------------
def load_eurosat():
    last = None
    for repo in ("blanchon/EuroSAT_RGB", "giswqs/EuroSAT_RGB", "timm/eurosat-rgb"):
        try:
            log(f"Loading dataset {repo} ...")
            ds = load_dataset(repo)
            parts = [ds[s] for s in ds.keys()]
            full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            return full, repo
        except Exception as e:  # noqa
            last = e
            log(f"  {repo} failed: {type(e).__name__}: {e}")
    raise SystemExit(f"All dataset sources failed. Last error: {last}")


if os.path.exists(CACHE):
    log(f"Loading cached features from {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    X, y = z["X"], z["y"]
    class_names = list(z["class_names"])
    source = str(z["source"])
    log(f"Cached features: {X.shape}, {len(class_names)} classes")
else:
    full, source = load_eurosat()
    cols = full.column_names
    img_col = next((c for c in ("image", "img", "rgb", "jpg") if c in cols), None)
    lbl_col = next((c for c in ("label", "labels", "class") if c in cols), None)
    try:
        class_names = full.features[lbl_col].names
    except Exception:
        class_names = sorted({str(x) for x in full[lbl_col]})
    log(f"Extracting features from {len(full)} images ...")
    t0 = time.time()
    X, y = [], []
    for i in range(len(full)):
        row = full[i]
        X.append(features_from_image(np.asarray(row[img_col].convert("RGB"))))
        y.append(int(row[lbl_col]))
        if (i + 1) % 5000 == 0:
            log(f"  {i + 1}/{len(full)}")
    X = np.vstack(X); y = np.asarray(y)
    np.savez_compressed(CACHE, X=X, y=y, class_names=np.array(class_names),
                        source=source)
    log(f"Features: {X.shape} in {time.time() - t0:.0f}s (cached to {CACHE})")

assert X.shape[1] == len(FEATURE_NAMES), \
    f"feature count {X.shape[1]} != names {len(FEATURE_NAMES)}"
n = X.shape[0]

# --------------------------------------------------------------------------
# 2. 5-fold stratified cross-validation across classifiers
# --------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "kappa": make_scorer(cohen_kappa_score),
}
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=SEED),
    "HistGradBoost": HistGradientBoostingClassifier(random_state=SEED),
    "kNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "LogReg": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000)),
    "Baseline(majority)": DummyClassifier(strategy="most_frequent"),
}

cv = {}
for name, clf in models.items():
    log(f"5-fold CV: {name} ...")
    tt = time.time()
    r = cross_validate(clf, X, y, cv=skf, scoring=scoring, n_jobs=1)
    cv[name] = {
        "accuracy_mean": float(np.mean(r["test_accuracy"])),
        "accuracy_std": float(np.std(r["test_accuracy"])),
        "f1_macro_mean": float(np.mean(r["test_f1_macro"])),
        "kappa_mean": float(np.mean(r["test_kappa"])),
        "kappa_std": float(np.std(r["test_kappa"])),
        "seconds": round(time.time() - tt, 1),
    }
    log(f"  {name}: acc={cv[name]['accuracy_mean']:.4f}"
        f"+/-{cv[name]['accuracy_std']:.4f} kappa={cv[name]['kappa_mean']:.4f}")

best_name = max((k for k in cv if not k.startswith("Baseline")),
                key=lambda k: cv[k]["accuracy_mean"])
log(f"Best model: {best_name}")

# --------------------------------------------------------------------------
# 3. Out-of-fold predictions for the best model -> confusion + per-class
#    (every sample predicted while held out; rigorous, uses all data)
# --------------------------------------------------------------------------
log(f"Out-of-fold predictions for {best_name} ...")
oof = cross_val_predict(models[best_name], X, y, cv=skf, n_jobs=1)
cm = confusion_matrix(y, oof)
report = classification_report(y, oof, target_names=class_names,
                               output_dict=True, zero_division=0)
oof_acc = float(accuracy_score(y, oof))
oof_kappa = float(cohen_kappa_score(y, oof))
oof_f1 = float(f1_score(y, oof, average="macro"))

# --------------------------------------------------------------------------
# 4. Feature importance (Random Forest, interpretable + fast)
# --------------------------------------------------------------------------
log("Random Forest feature importances ...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X, y)
imp = rf.feature_importances_
imp_order = np.argsort(imp)[::-1]
top_feats = [(FEATURE_NAMES[i], float(imp[i])) for i in imp_order[:15]]

# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (1) Confusion matrix (out-of-fold)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"{best_name} cross-validated confusion matrix\n"
             f"accuracy={oof_acc:.3f}  kappa={oof_kappa:.3f}")
th = cm.max() / 2
for r_ in range(cm.shape[0]):
    for c_ in range(cm.shape[1]):
        ax.text(c_, r_, cm[r_, c_], ha="center", va="center", fontsize=7,
                color="white" if cm[r_, c_] > th else "black")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_confusion.png"), dpi=160)

# (2) Per-class F1
f1s = [report[c]["f1-score"] for c in class_names]
fig, ax = plt.subplots(figsize=(9, 4))
order = np.argsort(f1s)
ax.barh([class_names[i] for i in order], [f1s[i] for i in order], color="teal")
ax.set_xlabel("F1 score"); ax.set_xlim(0, 1)
ax.set_title(f"{best_name} per-class F1 (cross-validated)")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_per_class_f1.png"), dpi=160)

# (3) Model comparison with CV std error bars
names = list(cv.keys())
means = [cv[k]["accuracy_mean"] for k in names]
stds = [cv[k]["accuracy_std"] for k in names]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(names)), means, yerr=stds, capsize=4, color="#4ea1ff")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("5-fold CV accuracy"); ax.set_ylim(0, 1)
ax.set_title("Classifier comparison on EuroSAT (identical features)")
for i, m in enumerate(means):
    ax.text(i, m + stds[i] + 0.01, f"{m:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_model_comparison.png"), dpi=160)

# (4) Feature importance (named)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([f for f, _ in reversed(top_feats)],
        [v for _, v in reversed(top_feats)], color="darkorange")
ax.set_xlabel("Random Forest importance")
ax.set_title("Top-15 most informative features")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_feature_importance.png"), dpi=160)

# --------------------------------------------------------------------------
# 6. Results + flat paper tokens
# --------------------------------------------------------------------------
water_classes = [c for c in class_names
                 if any(w in c.lower() for w in ("water", "river", "lake", "sea"))]
water_f1 = float(np.mean([report[c]["f1-score"] for c in water_classes])) \
    if water_classes else None

out = {
    "dataset": source, "n_images": int(n), "n_features": int(X.shape[1]),
    "n_classes": len(class_names), "classes": list(class_names),
    "evaluation": "5-fold stratified cross-validation",
    "cv_results": cv, "best_model": best_name,
    "best_oof_accuracy": oof_acc, "best_oof_kappa": oof_kappa,
    "best_oof_macro_f1": oof_f1,
    "water_classes": water_classes,
    "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in class_names},
    "confusion_matrix": cm.tolist(),
    "top_features": top_feats,
    "paper_tokens": {
        "dataset": source, "n_images": int(n), "n_features": int(X.shape[1]),
        "n_classes": len(class_names), "n_folds": 5,
        "best_model": best_name,
        "best_accuracy": round(cv[best_name]["accuracy_mean"], 4),
        "best_accuracy_std": round(cv[best_name]["accuracy_std"], 4),
        "best_kappa": round(cv[best_name]["kappa_mean"], 4),
        "best_kappa_std": round(cv[best_name]["kappa_std"], 4),
        "best_macro_f1": round(cv[best_name]["f1_macro_mean"], 4),
        "rf_accuracy": round(cv["RandomForest"]["accuracy_mean"], 4),
        "rf_kappa": round(cv["RandomForest"]["kappa_mean"], 4),
        "logreg_accuracy": round(cv["LogReg"]["accuracy_mean"], 4),
        "knn_accuracy": round(cv["kNN(k=5)"]["accuracy_mean"], 4),
        "baseline_accuracy": round(cv["Baseline(majority)"]["accuracy_mean"], 4),
        "water_f1": round(water_f1, 4) if water_f1 is not None else None,
        "top_feature_1": top_feats[0][0],
        "top_feature_2": top_feats[1][0],
        "top_feature_3": top_feats[2][0],
    },
}
with open(os.path.join(DATA, "results.json"), "w") as f:
    json.dump(out, f, indent=2)

log("=== DONE ===")
print(json.dumps(out["paper_tokens"], indent=2))
