"""
Solar Panel Soiling Detection - Main Entry Point
==================================================
Ties together data loading, model building, training, fine-tuning,
and evaluation.

Run from the project root:
    python main.py --data_dir data --epochs 15 --fine_tune_epochs 5 --out_dir outputs

See README.md for dataset setup instructions.
"""

import os
import warnings
import logging
import argparse

# ── Suppress noisy warnings before TF/PIL imports ─────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # suppress TF C++ INFO / WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # suppress oneDNN float-point notices
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("PIL").setLevel(logging.CRITICAL) # suppress PIL iCCP/sRGB warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# ──────────────────────────────────────────────────────────────────────────────

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.data_loader import load_datasets, get_class_weights
from src.model import build_model
from src.train import train_initial, train_fine_tune
from src.evaluate import evaluate_model, plot_history
from src.config import DEFAULT_EPOCHS, DEFAULT_FINE_TUNE_EPOCHS, DATA_DIR, OUTPUT_DIR



def main():
    parser = argparse.ArgumentParser(description="Solar Panel Soiling Detection")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                         help="Path to dataset root (must contain train/val/test folders)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                         help="Epochs for initial (frozen backbone) training")
    parser.add_argument("--fine_tune_epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS,
                         help="Epochs for optional fine-tuning stage (0 to skip)")
    parser.add_argument("--out_dir", type=str, default=OUTPUT_DIR,
                         help="Where to save the model, plots, and reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(args.data_dir)
    class_weights = get_class_weights(args.data_dir)
    print(f"Computed Class Weights: {class_weights}")

    print("Building model...")
    model, base_model = build_model()
    model.summary()

    print("Training (frozen backbone)...")
    history = train_initial(model, train_ds, val_ds, args.epochs, args.out_dir, class_weight=class_weights)
    plot_history(history, args.out_dir, tag="initial")

    if args.fine_tune_epochs > 0:
        print("Fine-tuning top layers of the backbone...")
        history_ft = train_fine_tune(
            model, base_model, train_ds, val_ds, args.fine_tune_epochs, args.out_dir, class_weight=class_weights
        )
        plot_history(history_ft, args.out_dir, tag="finetune")

    # Load best checkpoint for final evaluation and export
    best_ft_path = os.path.join(args.out_dir, "best_model_finetuned.keras")
    best_init_path = os.path.join(args.out_dir, "best_model.keras")
    
    best_checkpoint = best_ft_path if os.path.exists(best_ft_path) else best_init_path
    if os.path.exists(best_checkpoint):
        print(f"\nLoading best checkpoint from {best_checkpoint}...")
        model = tf.keras.models.load_model(best_checkpoint)

    print("Evaluating best model on test set...")
    evaluate_model(model, test_ds, args.out_dir)

    final_path = os.path.join(args.out_dir, "solar_soiling_final_model.keras")
    model.save(final_path)
    print(f"\nDone. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
