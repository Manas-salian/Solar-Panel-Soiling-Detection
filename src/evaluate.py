"""
Evaluation utilities: classification report, confusion matrix, and
training-curve plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import CLASS_NAMES


def evaluate_model(model, test_ds, out_dir):
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten().tolist())
        y_pred_probs.extend(preds.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    print("\nClassification Report:\n")
    print(report_text)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Solar Panel Soiling Detection")
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_history(history, out_dir, tag="initial"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title(f"Accuracy ({tag})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title(f"Loss ({tag})")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"training_curves_{tag}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {path}")
