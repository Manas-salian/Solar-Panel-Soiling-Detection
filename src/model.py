"""
Model definition: MobileNetV2 backbone (frozen initially) + a small
binary classification head. Includes a fine-tuning helper that unfreezes
the top layers of the backbone for a second training stage.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import IMG_SIZE
from src.data_loader import build_augmentation


def build_model():
    """Build the initial model with a frozen MobileNetV2 backbone."""
    augmentation = build_augmentation()

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, unfreeze_last_n=40, lr=2e-5):
    """Unfreeze the last N layers of the backbone and recompile with a
    lower learning rate, for a second (fine-tuning) training stage."""
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
