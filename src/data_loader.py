"""
Data loading utilities: builds train/val/test tf.data.Dataset objects
from an ImageFolder-style directory, plus a light augmentation layer.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import IMG_SIZE, BATCH_SIZE


def load_datasets(data_dir):
    """Load train/val/test datasets from data_dir/train, /val, /test."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


def get_class_weights(data_dir):
    """Compute balanced class weights to prevent majority-class bias."""
    train_clean = len(os.listdir(os.path.join(data_dir, "train", "clean")))
    train_dusty = len(os.listdir(os.path.join(data_dir, "train", "dusty")))
    total = train_clean + train_dusty
    weights = {
        0: (1 / train_clean) * (total / 2.0),
        1: (1 / train_dusty) * (total / 2.0),
    }
    return weights


def build_augmentation():
    """Robust data augmentation for various angles, lighting, and textures."""
    return models.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
