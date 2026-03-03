"""
train.py — Fine-tuning loop for the QR Quishing Detector.

Reads config.yaml, fine-tunes each model listed under ``models:``, saves
the best checkpoint (by F1), exports to ONNX and TFLite, and writes a
metrics summary CSV.

Usage
-----
    python src/train.py                   # uses config.yaml in cwd
    python src/train.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pathlib
import random
import time
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import build_dataloaders, UrlDataset, load_raw_data, balance_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple M1) > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run one evaluation pass and return loss, accuracy, F1."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n if n else float("nan")
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    onnx_path: pathlib.Path,
    max_length: int,
    device: torch.device,
) -> None:
    """Export the model to ONNX format."""
    import onnx  # noqa: F401  (checked at startup)

    model.eval()
    dummy_enc = tokenizer(
        ["https://example.com"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    input_ids = dummy_enc["input_ids"].to(device)
    attention_mask = dummy_enc["attention_mask"].to(device)

    dynamic_axes = {
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits": {0: "batch"},
    }
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    # token_type_ids presence is model-dependent
    has_tti = "token_type_ids" in dummy_enc
    if has_tti:
        token_type_ids = dummy_enc["token_type_ids"].to(device)
        dynamic_axes["token_type_ids"] = {0: "batch"}
        input_names.append("token_type_ids")
        torch_inputs = (input_ids, attention_mask, token_type_ids)
    else:
        torch_inputs = (input_ids, attention_mask)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("ONNX model saved → %s", onnx_path)


# ---------------------------------------------------------------------------
# TFLite export (via onnx → tf → tflite)
# ---------------------------------------------------------------------------

def export_tflite(onnx_path: pathlib.Path, tflite_path: pathlib.Path) -> None:
    """Convert an ONNX model to TFLite.

    Requires ``onnx-tf`` and ``tensorflow`` packages.  Emits a warning and
    skips silently if they are not installed — the ONNX file is always produced.
    """
    try:
        import onnx  # noqa: F401
        import onnx_tf  # noqa: F401
        import tensorflow as tf  # noqa: F401
    except ImportError:
        logger.warning(
            "onnx-tf / tensorflow not installed — skipping TFLite export. "
            "Run:  pip install onnx-tf tensorflow"
        )
        return

    from onnx_tf.backend import prepare

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_saved_model = tflite_path.parent / "tmp_saved_model"

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tmp_saved_model))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp_saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path.write_bytes(tflite_model)
    logger.info("TFLite model saved → %s", tflite_path)


# ---------------------------------------------------------------------------
# Training loop for a single model
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Fine-tune *model_name* and return best test metrics."""
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    epochs: int = int(train_cfg["epochs"])
    lr: float = float(train_cfg["learning_rate"])
    threshold: float = float(train_cfg["threshold"])
    ckpt_dir = pathlib.Path(out_cfg["checkpoint_dir"]) / model_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Training: %s ===", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_ckpt_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(**kwargs)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate_epoch(model, val_loader, device, loss_fn, threshold)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | %.1fs",
            epoch, epochs,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            time.time() - t0,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info("  ↳ New best checkpoint saved (val_f1=%.4f)", best_val_f1)

    # ── Load best checkpoint and evaluate on test set ──
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = evaluate_epoch(model, test_loader, device, loss_fn, threshold)
    logger.info(
        "Test results for %s: accuracy=%.4f, f1=%.4f",
        model_name, test_metrics["accuracy"], test_metrics["f1"],
    )

    # ── Export ──
    onnx_path = pathlib.Path(out_cfg["onnx_dir"]) / (model_name.replace("/", "_") + ".onnx")
    export_onnx(model, tokenizer, onnx_path, int(train_cfg["max_length"]), device)

    tflite_path = pathlib.Path(out_cfg["tflite_dir"]) / (model_name.replace("/", "_") + ".tflite")
    export_tflite(onnx_path, tflite_path)

    return {"model": model_name, **test_metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    set_seed(int(cfg["training"]["seed"]))
    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(config_path)

    results: list[dict] = []
    for model_name in cfg["models"]:
        metrics = train_model(model_name, cfg, train_loader, val_loader, test_loader, device)
        results.append(metrics)

    # ── Save metrics CSV ──
    metrics_path = pathlib.Path(cfg["output"]["metrics_csv"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(metrics_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    logger.info("Metrics saved → %s", metrics_path)

    # ── Print summary table ──
    col_w = 45
    header = f"{'Model':<{col_w}} {'Accuracy':>10} {'F1':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r['model']:<{col_w}} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
    print("=" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune URL classifiers")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
