# 513-deepfake-detection

> **Real or Fake?** A lightweight study of transfer learning for detecting
> AI-generated images, built for *Course 513 — Introduction to Deep Learning
> Applications and Theory* (Spring 2026).

Binary real-vs-fake image classification on the
[CIFAKE](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset)
benchmark (CIFAR-10 real images + Stable Diffusion 1.4 fakes). We compare:

1. **FFT + Logistic Regression** — radial frequency-spectrum baseline (65 params)
2. **Shallow CNN** — 4-block ConvNet trained from scratch (0.24M params)
3. **ResNet-18** — ImageNet-pretrained, fine-tuned end-to-end (11.2M params)
4. **EfficientNet-B0** — ImageNet-pretrained, fine-tuned end-to-end (4.0M params)

The whole experiment reproduces in **under 60 minutes on an Apple M4 laptop**
(MPS backend, no CUDA required).

---

## Results

Held-out test set, $n = 1{,}500$, balanced. Sorted by accuracy:

| Model                 |   Params | Test Acc | AUC    | F1     |
|-----------------------|---------:|---------:|-------:|-------:|
| Shallow CNN (scratch) |    241 K |   78.0 % | 0.9271 | 0.7259 |
| FFT + Logistic Reg.   |       65 |   81.1 % | 0.8850 | 0.8146 |
| EfficientNet-B0 (FT)  |    4.0 M |   84.7 % | 0.9767 | 0.8195 |
| **ResNet-18 (FT)**    | **11.2 M** | **90.0 %** | **0.9865** | **0.8904** |

Key findings: transfer learning wins decisively (+12 pts over the shallow CNN);
ResNet-18 *outperforms* the deeper EfficientNet-B0 at $96{\times}96$ input
resolution; and a 65-parameter FFT baseline beats the shallow CNN, confirming
that diffusion artifacts leave a detectable spectral fingerprint.

See [`report/main.tex`](report/main.tex) for the full write-up.

---

## Reproducing

### Prerequisites

```bash
# Python 3.10+
python3 -m pip install --user \
    torch torchvision scikit-learn datasets huggingface_hub \
    pillow tqdm numpy matplotlib certifi
```

### Run the full experiment

```bash
cd report/code
python3 run_experiments.py        # ~10 min on Apple M4 MPS
```

This downloads CIFAKE from HuggingFace (~100 MB), trains all four models, and
writes:

- `report/code/results/results.json` — full per-epoch metrics
- `report/figures/fig_training_curves.pdf`
- `report/figures/fig_confusion_matrix.pdf`
- `report/figures/fig_model_comparison.pdf`

### Build the report PDF

```bash
cd report
pdflatex main.tex && pdflatex main.tex   # twice for cross-references
```

---

## Repository layout

```
.
├── report/
│   ├── main.tex                  # NeurIPS-style writeup
│   ├── neurips_2022.sty
│   ├── figures/                  # generated figures (committed for paper compile)
│   └── code/
│       ├── data.py               # HuggingFace data loader + preprocessing
│       ├── models.py             # ShallowCNN / ResNet-18 / EfficientNet-B0
│       ├── train.py              # generic training & evaluation loop
│       ├── fft_baseline.py       # FFT + Logistic Regression baseline
│       ├── run_experiments.py    # main entry: trains all 4 models
│       ├── rerun_failed.py       # retry helper (handles HTTPS cert quirks)
│       └── results/results.json  # raw experiment output
└── generate_slides.py            # slide-deck generator (project proposal)
```

---

## Hardware

Trained on an Apple M4 MacBook (16 GB RAM) with the PyTorch MPS backend.
No CUDA / no cloud GPU. Total training time across all four models: ~10 minutes.

## Author

**Jianxu Shangguan** — `jxb1st@uw.edu`

## Acknowledgements

CIFAKE dataset: Bird & Lotfi, *IEEE Access* 2024. NeurIPS 2022 LaTeX style:
[neurips.cc](https://neurips.cc/Conferences/2022/PaperInformation/StyleFiles).
