"""
utils.py
========
Shared utilities: model compilation, callbacks, evaluation, and visualisation.
"""

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import callbacks

from configs.config import LABEL_SMOOTHING, LEARNING_RATE, IMG_SIZE, CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def set_mixed_precision(policy: str = "mixed_float16") -> None:
    """Enable Keras mixed-precision training."""
    tf.keras.mixed_precision.set_global_policy(policy)
    p = tf.keras.mixed_precision.global_policy()
    logger.info(
        "Mixed precision enabled – compute: %s, variable: %s",
        p.compute_dtype,
        p.variable_dtype,
    )


# ---------------------------------------------------------------------------
# Model compilation helpers
# ---------------------------------------------------------------------------

def compile_model_single(model: tf.keras.Model, lr: float = LEARNING_RATE) -> None:
    """Compile model for single-output training (Stage 1)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def compile_model_multi(model: tf.keras.Model, lr: float = LEARNING_RATE) -> None:
    """Compile model for multi-output training (Stages 2 & 3)."""
    loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "main_output": loss_fn,
            "color_output": loss_fn,
            "freq_output": loss_fn,
        },
        loss_weights={
            "main_output": 1.0,
            "color_output": 0.3,
            "freq_output": 0.3,
        },
        metrics={
            "main_output": ["accuracy", keras.metrics.AUC(name="auc")],
            "color_output": ["accuracy"],
            "freq_output": ["accuracy"],
        },
    )


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------

def get_callbacks(monitor: str = "val_loss") -> list:
    """
    Return standard Keras callbacks.

    Includes:
      - EarlyStopping  (patience=6, restores best weights)
      - ReduceLROnPlateau (factor=0.5, patience=3)
    """
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: tf.keras.Model, path: str) -> None:
    """Save the full Keras model (architecture + weights) to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    logger.info("Model saved → %s", path)


def load_model(path: str) -> tf.keras.Model:
    """Load a previously saved Keras model from *path*."""
    logger.info("Loading model from %s", path)
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple:
    """
    Run inference on *dataset* and return (predictions, true_labels).

    Works for both single-output and multi-output models (main_output only).
    """
    all_preds, all_labels = [], []
    for images, labels in dataset:
        outputs = model(images, training=False)
        preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        all_preds.extend(preds.numpy().flatten())
        all_labels.extend(labels.numpy().flatten())
    return np.array(all_preds), np.array(all_labels)


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Find the classification threshold that maximises (TPR - FPR) on the ROC
    curve (Youden's J statistic).
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float | None = None,
    val_dataset: tf.data.Dataset | None = None,
) -> dict:
    """
    Evaluate *model* on *dataset* (intended to be the TEST set) and return a
    metrics dictionary.

    Threshold selection methodology
    --------------------------------
    The classification threshold must be chosen using the VALIDATION set,
    never the test set, to avoid leaking test information into the decision
    boundary. Pass one of:
      - `threshold`     : a fixed value (e.g. CLASSIFICATION_THRESHOLD from
                           configs/config.py) to skip selection entirely, or
      - `val_dataset`    : a validation ``tf.data.Dataset`` — the optimal
                           threshold is computed on it via Youden's J
                           statistic and then applied to `dataset` (test).
    If neither is given, this raises an error rather than silently picking
    the threshold on the test set itself.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc_roc, threshold,
                    confusion_matrix, classification_report
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if threshold is None:
        if val_dataset is None:
            raise ValueError(
                "evaluate_model: no threshold provided and no val_dataset "
                "given. Pass a fixed `threshold` (e.g. CLASSIFICATION_"
                "THRESHOLD from configs/config.py) or a `val_dataset` to "
                "select the threshold on — never select it on the test set."
            )
        y_val_pred, y_val_true = get_predictions(model, val_dataset)
        threshold = find_optimal_threshold(y_val_true, y_val_pred)
        logger.info("Optimal threshold selected on validation set: %.4f", threshold)

    y_pred, y_true = get_predictions(model, dataset)

    y_pred_bin = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred)

    results = {
        "threshold": threshold,
        "accuracy": float(np.mean(y_pred_bin == y_true.astype(int))),
        "precision": precision_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "recall": recall_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "f1": f1_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "auc_roc": auc,
        "confusion_matrix": confusion_matrix(y_true.astype(int), y_pred_bin),
        "classification_report": classification_report(
            y_true.astype(int), y_pred_bin, target_names=["Real", "AI"]
        ),
    }
    return results


def analyse_attention_weights(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> dict:
    """
    Extract and summarise the attention weights from the fusion layer.

    Returns
    -------
    dict with per-branch mean and std of attention weights.
    """
    attn_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("attention_weights").output,
    )
    all_attn = []
    for images, _ in dataset:
        aw = attn_model(images, training=False).numpy()
        all_attn.extend(aw)
    all_attn = np.array(all_attn)

    return {
        "spatial": {"mean": float(all_attn[:, 0].mean()), "std": float(all_attn[:, 0].std())},
        "color":   {"mean": float(all_attn[:, 1].mean()), "std": float(all_attn[:, 1].std())},
        "freq":    {"mean": float(all_attn[:, 2].mean()), "std": float(all_attn[:, 2].std())},
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(histories: list, save_path: str | None = None) -> None:
    """
    Plot combined loss and accuracy curves across multiple training histories.

    Parameters
    ----------
    histories : list of keras.callbacks.History
        One per training stage.
    save_path : str | None
        If provided, the figure is saved to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_loss, all_val_loss = [], []
    all_acc, all_val_acc = [], []
    stage_boundaries = []

    for hist in histories:
        all_loss.extend(hist.history["loss"])
        all_val_loss.extend(hist.history["val_loss"])
        stage_boundaries.append(len(all_loss))

        acc_keys = [k for k in hist.history if "accuracy" in k and "val" not in k]
        val_acc_keys = [k for k in hist.history if "accuracy" in k and "val" in k]
        if acc_keys:
            all_acc.extend(hist.history[acc_keys[0]])
        if val_acc_keys:
            all_val_acc.extend(hist.history[val_acc_keys[0]])

    epochs = range(1, len(all_loss) + 1)

    axes[0].plot(epochs, all_loss, "b-", label="Train Loss")
    axes[0].plot(epochs, all_val_loss, "r-", label="Val Loss")
    axes[0].set_title("Loss Across All Stages")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if all_acc:
        axes[1].plot(range(1, len(all_acc) + 1), all_acc, "b-", label="Train Acc")
    if all_val_acc:
        axes[1].plot(range(1, len(all_val_acc) + 1), all_val_acc, "r-", label="Val Acc")
    axes[1].set_title("Accuracy Across All Stages")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    for ax in axes:
        for boundary in stage_boundaries[:-1]:
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Training curves saved → %s", save_path)
    plt.show()


def predict_single_image(
    model: tf.keras.Model, image_path: str, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict:
    """
    Run inference on a single image file and return a result dictionary.

    Parameters
    ----------
    model       : Loaded FDCS-Net V4 model.
    image_path  : Path to the image file.
    threshold   : Classification threshold (default 0.5).

    Returns
    -------
    dict with keys: prediction, confidence, label
    """
    from data_preprocessing import parse_image

    img, _ = parse_image(image_path, 0)
    img = tf.expand_dims(img, 0)                    # add batch dim

    outputs = model(img, training=False)
    prob = float(outputs[0].numpy() if isinstance(outputs, (list, tuple)) else outputs.numpy())

    return {
        "prediction": prob,
        "confidence": max(prob, 1 - prob),
        "label": "AI-Generated" if prob >= threshold else "Real",
    }