"""
Evaluation utilities: batched prediction, metric computation, and plots
(confusion matrix, training curves).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.config import CLASS_INFO  # noqa: E402

# Chart palette (see dataviz reference palette): one hue per job.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES_1 = "#2a78d6"   # train
SERIES_2 = "#eb6834"   # val
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_loader(model, loader, device, use_amp: bool):
    """Return (probs [N, K] float32, labels [N] int64)."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def compute_metrics(probs: np.ndarray, labels: np.ndarray, class_names) -> dict:
    preds = probs.argmax(axis=1)
    k = len(class_names)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=range(k), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=range(k))
    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "top2_accuracy": float(top_k_accuracy_score(labels, probs, k=2, labels=range(k))),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "roc_auc_ovr_macro": auc,
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(labels)),
    }


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix (test set)"):
    cm = np.asarray(cm)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [CLASS_INFO.get(c, {}).get("label", c) for c in class_names]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#fcfcfb")
    ax.imshow(row_pct, cmap=BLUES_CMAP, vmin=0, vmax=1)
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            color = "#ffffff" if row_pct[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color=color, linespacing=1.3)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color=INK_2)
    ax.set_yticklabels(labels, fontsize=9, color=INK_2)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("Actual", color=INK_2)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_history(history: dict, path, title: str, warmup_epochs: int = 0, best_epoch: int | None = None):
    """history keys: train_loss, val_loss, train_acc, val_acc, val_macro_f1 (lists per epoch)."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#fcfcfb")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=SERIES_1, lw=2, label="train")
    ax.plot(epochs, history["val_loss"], color=SERIES_2, lw=2, label="validation")
    ax.set_title("Cross-entropy loss", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=INK_2)
    _style_axes(ax)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=SERIES_1, lw=2, label="train accuracy")
    ax.plot(epochs, history["val_acc"], color=SERIES_2, lw=2, label="validation accuracy")
    ax.plot(epochs, history["val_macro_f1"], color=SERIES_2, lw=2, ls="--", label="validation macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy / macro-F1", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=INK_2)
    _style_axes(ax)

    for ax in axes:
        if warmup_epochs:
            ax.axvline(warmup_epochs + 0.5, color=MUTED, lw=1, ls=":")
            ax.text(warmup_epochs + 0.6, ax.get_ylim()[1], "head-only | full fine-tune",
                    fontsize=8, color=MUTED, va="top")
        if best_epoch:
            ax.axvline(best_epoch, color=GRID, lw=6, alpha=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
