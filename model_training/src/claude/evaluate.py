"""
evaluate.py — Model evaluation and comparison for the QR Quishing Detector.

Loads saved checkpoints (or a results CSV produced by train.py) and
computes a full set of classification metrics including baseline
comparisons.

Usage
-----
    # Compare all models using the metrics CSV written by train.py:
    python src/evaluate.py --metrics results/metrics.csv

    # Re-evaluate a single checkpoint on the test split:
    python src/evaluate.py --checkpoint checkpoints/distilbert-base-uncased/best_model.pt \
                            --model distilbert-base-uncased \
                            --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 or 1).
    y_pred:
        Predicted labels (0 or 1).

    Returns
    -------
    Dictionary with keys: tp, tn, fp, fn, accuracy, precision, recall,
    f1, specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "specificity": round(float(specificity), 6),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_random(y_true: list[int], seed: int = 42) -> dict[str, float]:
    """Random classifier baseline (50 % probability per class)."""
    rng = np.random.default_rng(seed)
    y_pred = rng.integers(0, 2, size=len(y_true)).tolist()
    return compute_metrics(y_true, y_pred)


def baseline_majority(y_true: list[int]) -> dict[str, float]:
    """Always-predict-majority-class baseline."""
    majority = int(np.bincount(y_true).argmax())
    y_pred = [majority] * len(y_true)
    return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: pathlib.Path,
    model_name: str,
    test_loader: DataLoader,
    config: dict,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Load a checkpoint and evaluate it against the test set.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``best_model.pt`` saved by train.py.
    model_name:
        HuggingFace model identifier (must match the checkpoint).
    test_loader:
        Test DataLoader.
    config:
        Parsed config.yaml dict.
    device:
        Torch device; auto-detected if ``None``.

    Returns
    -------
    Metrics dictionary (same shape as ``compute_metrics``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = float(config["training"]["threshold"])

    logger.info("Loading model '%s' from %s …", model_name, checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    logger.info("Evaluated '%s': %s", model_name, metrics)
    return {"model": model_name, **metrics}


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_comparison_table(rows: list[dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not rows:
        print("No results to display.")
        return

    cols = ["model", "accuracy", "precision", "recall", "f1", "specificity", "tp", "tn", "fp", "fn"]
    # Only include columns that exist in the data
    cols = [c for c in cols if c in rows[0]]

    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = " | ".join(f"{c:<{col_widths[c]}}" for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)

    print("\n" + sep)
    print(header)
    print(sep)
    for row in rows:
        line = " | ".join(f"{str(row.get(c, '')):<{col_widths[c]}}" for c in cols)
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare URL classifier models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Path to metrics CSV written by train.py (skips re-evaluation)",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to a single .pt checkpoint")
    parser.add_argument("--model", default=None, help="Model name matching --checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    results: list[dict] = []

    if args.metrics:
        # ── Load pre-computed metrics from CSV ──
        metrics_path = pathlib.Path(args.metrics)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        df = pd.read_csv(metrics_path)
        results = df.to_dict(orient="records")
        logger.info("Loaded %d model results from %s", len(results), metrics_path)

    elif args.checkpoint and args.model:
        # ── Re-evaluate a specific checkpoint ──
        _, _, test_loader = build_dataloaders(args.config)
        row = evaluate_checkpoint(
            pathlib.Path(args.checkpoint), args.model, test_loader, cfg
        )
        results.append(row)

    else:
        # ── Re-evaluate all checkpoints found in checkpoint_dir ──
        ckpt_base = pathlib.Path(cfg["output"]["checkpoint_dir"])
        _, _, test_loader = build_dataloaders(args.config)
        for model_name in cfg["models"]:
            ckpt = ckpt_base / model_name.replace("/", "_") / "best_model.pt"
            if ckpt.exists():
                row = evaluate_checkpoint(ckpt, model_name, test_loader, cfg)
                results.append(row)
            else:
                logger.warning("Checkpoint not found, skipping: %s", ckpt)

    if not results:
        logger.error("No results to report.")
        return

    # ── Add baselines using a dummy label list if available ──
    # Baselines only make sense when we have ground-truth from evaluation
    if "tp" in results[0]:
        # Reconstruct approximate y_true from confusion matrix counts
        first = results[0]
        n_total = first["tp"] + first["tn"] + first["fp"] + first["fn"]
        n_pos = first["tp"] + first["fn"]
        y_approx = [1] * n_pos + [0] * (n_total - n_pos)

        rnd_metrics = baseline_random(y_approx)
        maj_metrics = baseline_majority(y_approx)
        results.append({"model": "BASELINE: random", **rnd_metrics})
        results.append({"model": "BASELINE: majority", **maj_metrics})

    print_comparison_table(results)

    # ── Save enriched CSV ──
    out_path = pathlib.Path(cfg["output"]["metrics_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
