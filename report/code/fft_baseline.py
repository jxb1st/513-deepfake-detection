"""Classical frequency-domain baseline: FFT magnitude features + logistic regression.

Motivation: GAN- and diffusion-generated images leave characteristic
spectral fingerprints. A linear model over the radial FFT magnitude
spectrum is a strong sanity check and tells us how much of the signal
is in the frequency domain vs. learned features.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

FEAT_DIM = 64  # length of radial spectrum profile


def radial_profile(mag: np.ndarray, n_bins: int = FEAT_DIM) -> np.ndarray:
    """Radial mean of a 2-D FFT magnitude spectrum."""
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(np.max(r))
    bins = np.linspace(0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    sums = np.bincount(idx, weights=mag.ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins).clip(min=1)
    return sums / counts


def image_to_features(pil_img: Image.Image, size: int = 128) -> np.ndarray:
    img = pil_img.convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr - arr.mean()
    f = np.fft.fft2(arr)
    f = np.fft.fftshift(f)
    mag = np.log1p(np.abs(f))
    return radial_profile(mag)


def extract_features_from_hf(hf_ds, image_key: str, label_key: str,
                              max_n: int | None = None):
    n = len(hf_ds) if max_n is None else min(max_n, len(hf_ds))
    X = np.zeros((n, FEAT_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        row = hf_ds[i]
        img = row[image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        X[i] = image_to_features(img)
        y[i] = int(row[label_key])
    return X, y


def run_fft_baseline(bundle):
    """Run FFT + LR on the same data the deep models see.

    Returns a dict of metrics matching the deep-model report schema.
    """
    train_t = bundle.train_loader.dataset
    val_t = bundle.val_loader.dataset
    test_t = bundle.test_loader.dataset

    print("[fft] extracting features...")
    X_tr, y_tr = extract_features_from_hf(
        train_t.ds, train_t.image_key, train_t.label_key)
    X_va, y_va = extract_features_from_hf(
        val_t.ds, val_t.image_key, val_t.label_key)
    X_te, y_te = extract_features_from_hf(
        test_t.ds, test_t.image_key, test_t.label_key)
    print(f"[fft] features: train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    def _evaluate(X, y):
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "acc": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds)),
            "auc": float(roc_auc_score(y, probs)) if len(set(y)) > 1
                    else float("nan"),
        }

    val_m = _evaluate(X_va, y_va)
    test_m = _evaluate(X_te, y_te)
    print(f"[fft] val: {val_m}   test: {test_m}")

    from sklearn.metrics import confusion_matrix
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_te, preds).tolist()

    return {
        "model_name": "FFT + Logistic Regression",
        "n_params": int(X_tr.shape[1] + 1),
        "epochs": 0,
        "history": [],
        "test_acc": test_m["acc"],
        "test_auc": test_m["auc"],
        "test_f1": test_m["f1"],
        "test_confusion": cm,
        "train_minutes": 0.0,
    }
