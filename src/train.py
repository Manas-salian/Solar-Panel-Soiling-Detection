"""
Training routines: initial training (frozen backbone) and an optional
fine-tuning stage, both with early stopping and checkpointing.
"""

import os
from tensorflow.keras import callbacks

from src.model import unfreeze_for_fine_tuning


def train_initial(model, train_ds, val_ds, epochs, out_dir, class_weight=None):
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb,
        class_weight=class_weight,
    )
    return history


def train_fine_tune(model, base_model, train_ds, val_ds, epochs, out_dir, class_weight=None):
    model = unfreeze_for_fine_tuning(model, base_model)
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_model_finetuned.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb,
        class_weight=class_weight,
    )
    return history_ft
