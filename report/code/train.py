"""Generic training and evaluation loops."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_auc: float
    seconds: float


@dataclass
class RunResult:
    model_name: str
    n_params: int
    epochs: int
    history: list = field(default_factory=list)
    test_acc: float = 0.0
    test_auc: float = 0.0
    test_f1: float = 0.0
    test_confusion: list = field(default_factory=list)
    train_minutes: float = 0.0


def _epoch(model, loader, device, *, optim=None, criterion=None):
    """One pass; if optim is given we train, else eval. Returns metrics."""
    is_train = optim is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            preds = logits.argmax(1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            total += y.size(0)
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "probs": probs,
        "labels": labels,
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> RunResult:
    """Fit one model and return a packaged result."""
    from models import count_params

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    result = RunResult(model_name=model_name,
                       n_params=count_params(model), epochs=epochs)
    t_start = time.time()

    print(f"\n=== {model_name} ({result.n_params/1e6:.2f}M params) ===")
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        tr = _epoch(model, train_loader, device, optim=optim, criterion=criterion)
        vl = _epoch(model, val_loader, device, optim=None, criterion=criterion)
        scheduler.step()
        log = EpochLog(epoch=ep,
                       train_loss=tr["loss"], train_acc=tr["acc"],
                       val_loss=vl["loss"], val_acc=vl["acc"],
                       val_auc=vl["auc"], seconds=time.time() - t_ep)
        result.history.append(asdict(log))
        print(f"  epoch {ep}/{epochs}: "
              f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f}  "
              f"val_loss={vl['loss']:.4f} val_acc={vl['acc']:.4f} "
              f"val_auc={vl['auc']:.4f}  [{log.seconds:.1f}s]")

    te = _epoch(model, test_loader, device, optim=None, criterion=criterion)
    preds = (te["probs"] >= 0.5).astype(int)
    result.test_acc = float(accuracy_score(te["labels"], preds))
    result.test_auc = float(te["auc"])
    result.test_f1 = float(f1_score(te["labels"], preds))
    result.test_confusion = confusion_matrix(te["labels"], preds).tolist()
    result.train_minutes = (time.time() - t_start) / 60.0
    print(f"  TEST: acc={result.test_acc:.4f} auc={result.test_auc:.4f} "
          f"f1={result.test_f1:.4f}  [total {result.train_minutes:.1f} min]")
    return result
