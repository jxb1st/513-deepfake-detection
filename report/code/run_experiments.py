"""End-to-end: load data, train all models, save metrics + figures.

Run:
    cd report/code && python3 run_experiments.py

Outputs:
    report/code/results/results.json  -- full metrics for all models
    report/figures/fig_training_curves.pdf
    report/figures/fig_confusion_matrix.pdf
    report/figures/fig_model_comparison.pdf
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

# Fix SSL cert path BEFORE importing libraries that download weights.
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
import ssl
try:
    ssl._create_default_https_context = ssl.create_default_context
except Exception:
    pass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load_data, set_seed, get_device
from models import REGISTRY
from train import train_model
from fft_baseline import run_fft_baseline

RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def save_results(results: dict) -> None:
    out = os.path.join(RESULTS_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[saved] {out}")


def plot_training_curves(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": 0.7, "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })
    colors = {"ShallowCNN": "#5A6B7C", "ResNet-18": "#2BB3A3",
              "EfficientNet-B0": "#16365C",
              "MobileNet-V3-Small": "#E07A5F"}
    markers = {"ShallowCNN": "o", "ResNet-18": "s",
               "EfficientNet-B0": "^", "MobileNet-V3-Small": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for name, res in results["models"].items():
        if not res.get("history"):
            continue
        h = res["history"]
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["val_acc"] * 100 for e in h],
                     marker=markers.get(name, "o"), ms=3.5, lw=1.3,
                     color=colors.get(name, "k"), label=name)
        axes[1].plot(ep, [e["val_loss"] for e in h],
                     marker=markers.get(name, "o"), ms=3.5, lw=1.3,
                     color=colors.get(name, "k"), label=name)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].grid(True, ls="--"); axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("(a) Validation accuracy")

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation loss")
    axes[1].grid(True, ls="--"); axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("(b) Validation loss")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_training_curves.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[plot] {out}")


def plot_confusion(results: dict, best_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "savefig.bbox": "tight"})
    cm = np.array(results["models"][best_name]["test_confusion"])
    label_names = results.get("label_names", ["class 0", "class 1"])

    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    cmap = LinearSegmentedColormap.from_list("teal", ["#EAF6F4", "#2BB3A3", "#16365C"])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        c = "white" if v > cm.max() * 0.4 else "#16365C"
        ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                color=c, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix --- {best_name}")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_confusion_matrix.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[plot] {out}")


def plot_model_comparison(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "savefig.bbox": "tight",
    })
    # Only models with valid metrics; sort so that bars read left-to-right
    # from weakest to strongest by accuracy.
    valid = [(n, r) for n, r in results["models"].items()
             if isinstance(r, dict) and "test_acc" in r]
    if not valid:
        print("[plot] no valid models to compare; skipping")
        return
    valid.sort(key=lambda kv: kv[1]["test_acc"])
    names = [n for n, _ in valid]
    short = [n.replace("EfficientNet-B0", "EffNet-B0")
              .replace("ShallowCNN", "Shallow CNN") for n in names]
    accs = [r["test_acc"] * 100 for _, r in valid]
    aucs = [r["test_auc"] * 100 for _, r in valid]
    f1s = [r["test_f1"] * 100 for _, r in valid]

    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    b1 = ax.bar(x - w, accs, w, color="#16365C", label="Accuracy")
    b2 = ax.bar(x,     aucs, w, color="#2BB3A3", label="AUC")
    b3 = ax.bar(x + w, f1s,  w, color="#E07A5F", label="F1")
    for bars in (b1, b2, b3):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.0,
                    f"{rect.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("Test metric (\\%)")
    ax.set_ylim(60, 105)
    ax.grid(True, axis="y", ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Test-set performance across models")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_model_comparison.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"[plot] {out}")


def main():
    set_seed(513)
    device = get_device()
    print(f"[env] device = {device}")

    # Dataset size budget (tuned so total runtime on Apple Silicon ~30-50 min)
    bundle = load_data(n_train=4000, n_val=800, n_test=1500, batch_size=64)
    print(f"[data] {bundle.name}: train={bundle.n_train} val={bundle.n_val} "
          f"test={bundle.n_test}  labels={bundle.label_names}")

    results = {
        "dataset": bundle.name,
        "n_train": bundle.n_train,
        "n_val": bundle.n_val,
        "n_test": bundle.n_test,
        "label_names": list(bundle.label_names),
        "device": str(device),
        "models": {},
    }

    # Train deep models
    deep_models_to_run = [
        ("ShallowCNN",       5, 1e-3),
        ("ResNet-18",        5, 3e-4),
        ("EfficientNet-B0",  5, 3e-4),
    ]
    for name, epochs, lr in deep_models_to_run:
        try:
            model = REGISTRY[name]()
            r = train_model(name, model, bundle.train_loader,
                            bundle.val_loader, bundle.test_loader,
                            device, epochs=epochs, lr=lr)
            results["models"][name] = asdict(r)
            save_results(results)
        except Exception as e:
            print(f"[error] {name} failed: {e}")
            results["models"][name] = {"error": str(e)}
            save_results(results)

    # FFT baseline (classical)
    try:
        fft = run_fft_baseline(bundle)
        results["models"]["FFT + LR"] = fft
    except Exception as e:
        print(f"[error] FFT baseline failed: {e}")

    save_results(results)

    # Pick best by test accuracy
    valid = {n: r for n, r in results["models"].items()
             if isinstance(r, dict) and "test_acc" in r}
    if valid:
        best_name = max(valid, key=lambda n: valid[n]["test_acc"])
        print(f"\n[summary] best model = {best_name} "
              f"(test_acc={valid[best_name]['test_acc']:.4f})")
        plot_training_curves(results)
        plot_confusion(results, best_name)
        plot_model_comparison(results)

    print("\n[done]")


if __name__ == "__main__":
    main()
