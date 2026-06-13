"""Re-run only the models that failed in the first pass (SSL cert issue).

Preserves successful entries (ShallowCNN, FFT + LR) in results.json,
re-trains failed transfer-learning models, then regenerates figures.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

# Fix SSL cert path BEFORE PyTorch tries to download pretrained weights.
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load_data, set_seed, get_device
from models import REGISTRY
from train import train_model
from run_experiments import (
    plot_training_curves, plot_confusion, plot_model_comparison,
    RESULTS_DIR, save_results,
)


def main():
    set_seed(513)
    device = get_device()
    print(f"[env] device = {device}")

    # Load existing results
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path) as f:
        results = json.load(f)
    print(f"[load] existing results: {list(results['models'].keys())}")

    # Identify failed models (those with 'error' field instead of 'test_acc')
    failed = [n for n, r in results["models"].items()
              if isinstance(r, dict) and "test_acc" not in r]
    print(f"[load] failed models to re-run: {failed}")

    if not failed:
        print("[load] nothing to re-run; regenerating plots only")
    else:
        # Reload data with the SAME seed to match the original split exactly.
        bundle = load_data(n_train=4000, n_val=800, n_test=1500, batch_size=64)
        print(f"[data] {bundle.name}: train={bundle.n_train} "
              f"val={bundle.n_val} test={bundle.n_test}")

        # Same hyperparameters as the original run
        hp = {"ResNet-18": (5, 3e-4), "EfficientNet-B0": (5, 3e-4)}
        for name in failed:
            if name not in REGISTRY:
                print(f"[skip] {name} not in registry"); continue
            epochs, lr = hp.get(name, (5, 3e-4))
            try:
                model = REGISTRY[name]()
                r = train_model(name, model, bundle.train_loader,
                                bundle.val_loader, bundle.test_loader,
                                device, epochs=epochs, lr=lr)
                results["models"][name] = asdict(r)
                save_results(results)
            except Exception as e:
                print(f"[error] {name} failed AGAIN: {e}")
                results["models"][name] = {"error": str(e)}
                save_results(results)

    # Regenerate plots with whatever we have
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
