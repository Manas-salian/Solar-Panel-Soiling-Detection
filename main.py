"""
Solar Panel Fault & Soiling Detection - training entry point
=============================================================
Trains one or more pretrained backbones on the 6-class dataset, evaluates each
on the held-out test split, picks the best by *validation* macro-F1 (so the test
set never influences model selection), and exports it for the app.

Run from the project root:
    python scripts/prepare_dataset.py --raw_dir Faulty_solar_panel --out_dir data
    python main.py                                   # default 3-backbone comparison
    python main.py --backbones efficientnet_v2_s     # single model, faster

Outputs (in --out_dir):
    runs/<backbone>/{best_model.pt, metrics.json, history.json, confusion_matrix.png, training_curves.png}
    comparison.json, comparison.md      leaderboard of all backbones
    metrics.json, confusion_matrix.png, training_curves.png   copied from the winner
    final_model.pt                       self-describing checkpoint used by app.py
"""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import torch

from src.config import (
    BATCH_SIZE, CLASS_NAMES, COMPARISON_PATH, DATA_DIR, DEFAULT_BACKBONES, DEFAULT_EPOCHS,
    DEFAULT_WARMUP_EPOCHS, EARLY_STOPPING_PATIENCE, FINAL_MODEL_PATH, IMAGENET_MEAN,
    IMAGENET_STD, IMG_SIZE, NUM_WORKERS, OUTPUT_DIR, SEED,
)
from src.data import build_dataloaders
from src.evaluate import compute_metrics, plot_confusion_matrix, predict_loader, save_json
from src.model import SUPPORTED_BACKBONES, count_parameters, measure_latency_ms
from src.train import train_backbone


def parse_args():
    ap = argparse.ArgumentParser(description="Train and compare backbones for solar panel fault detection.")
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES),
                    help=f"Comma-separated subset of {SUPPORTED_BACKBONES}")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max fine-tune epochs (early stopping applies)")
    ap.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS, help="Head-only epochs before unfreezing")
    ap.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Backbones: {backbones}")

    loaders, train_counts = build_dataloaders(args.data_dir, args.img_size, args.batch_size, args.num_workers)
    print("Train class counts:", dict(zip(CLASS_NAMES, map(int, train_counts))))

    results = []
    for backbone in backbones:
        print(f"\n=== {backbone} ===")
        run_dir = out_dir / "runs" / backbone
        r = train_backbone(
            backbone, loaders, train_counts, CLASS_NAMES,
            epochs=args.epochs, warmup_epochs=args.warmup_epochs, patience=args.patience,
            seed=args.seed, device=device, run_dir=run_dir,
        )
        model = r["model"]

        probs, labels = predict_loader(model, loaders["test"], device, use_amp=device.type == "cuda")
        test_metrics = compute_metrics(probs, labels, CLASS_NAMES)
        plot_confusion_matrix(test_metrics["confusion_matrix"], CLASS_NAMES, run_dir / "confusion_matrix.png",
                              title=f"{backbone}: confusion matrix (test set)")

        gpu_ms = measure_latency_ms(model, args.img_size, device) if device.type == "cuda" else None
        cpu_ms = measure_latency_ms(model.to("cpu"), args.img_size, torch.device("cpu"), n_iters=15)
        model.to(device)

        summary = {
            "backbone": backbone,
            "params_millions": round(count_parameters(model) / 1e6, 2),
            "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "train_time_min": round(r["train_time_sec"] / 60, 1),
            "val_macro_f1": round(r["best_val_macro_f1"], 4),
            "val_accuracy": round(r["best_val_metrics"]["accuracy"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_roc_auc": None if test_metrics["roc_auc_ovr_macro"] is None else round(test_metrics["roc_auc_ovr_macro"], 4),
            "latency_gpu_ms": None if gpu_ms is None else round(gpu_ms, 1),
            "latency_cpu_ms": round(cpu_ms, 1),
        }
        results.append({**summary, "_state": r["best_state"], "_test_metrics": test_metrics,
                        "_val_metrics": r["best_val_metrics"], "_history": r["history"]})

        save_json({**test_metrics, "backbone": backbone, "split": "test"}, run_dir / "metrics.json")
        save_json(r["history"], run_dir / "history.json")
        print(f"  test: acc {test_metrics['accuracy']:.3f}  macro-F1 {test_metrics['macro_f1']:.3f}  "
              f"| {summary['params_millions']} M params, {summary['latency_cpu_ms']} ms/img CPU")

    # ---- Select by validation macro-F1 (never by test) --------------------- #
    best = max(results, key=lambda r: (r["val_macro_f1"], -r["params_millions"]))
    print(f"\nSelected backbone: {best['backbone']} (val macro-F1 {best['val_macro_f1']:.3f})")

    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    comparison = {
        "selected": best["backbone"],
        "selection_rule": "highest validation macro-F1 (ties -> fewer parameters)",
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(args).items()},
        "device": str(device),
        "results": public,
    }
    save_json(comparison, out_dir / COMPARISON_PATH.name)
    with open(out_dir / "comparison.md", "w") as f:
        cols = ["backbone", "params_millions", "val_macro_f1", "val_accuracy", "test_accuracy",
                "test_macro_f1", "test_roc_auc", "latency_cpu_ms", "latency_gpu_ms", "best_epoch", "train_time_min"]
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in public:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

    run_dir = out_dir / "runs" / best["backbone"]
    for name in ("confusion_matrix.png", "training_curves.png"):
        shutil.copy(run_dir / name, out_dir / name)
    save_json({**best["_test_metrics"], "backbone": best["backbone"], "split": "test",
               "val_macro_f1": best["val_macro_f1"], "val_accuracy": best["val_accuracy"]},
              out_dir / "metrics.json")

    torch.save({
        "format_version": 1,
        "backbone": best["backbone"],
        "class_names": CLASS_NAMES,
        "img_size": args.img_size,
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "trained_at": comparison["trained_at"],
        "params_millions": best["params_millions"],
        "val_macro_f1": best["val_macro_f1"],
        "test_accuracy": best["test_accuracy"],
        "test_macro_f1": best["test_macro_f1"],
        "latency_cpu_ms": best["latency_cpu_ms"],
        "state_dict": best["_state"],
    }, out_dir / FINAL_MODEL_PATH.name)

    print("\n" + open(out_dir / "comparison.md").read())
    print(f"Final model -> {out_dir / FINAL_MODEL_PATH.name}")


if __name__ == "__main__":
    main()
