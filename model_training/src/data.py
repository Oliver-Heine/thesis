"""
data.py — Dataset loading and preprocessing for the QR Quishing Detector.

Expects a CSV file with at minimum two columns:
  - url    : raw URL string
  - result : integer label (0 = benign, 1 = malicious)

Usage
-----
    from data import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders("config.yaml")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"^(https?://)"               # required scheme
    r"([A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)$",
    re.IGNORECASE,
)
_MAX_URL_LEN = 2048


# ---------------------------------------------------------------------------
# URL preprocessing
# ---------------------------------------------------------------------------

def preprocess_url(url: str) -> Optional[str]:
    """Lowercase and strip a URL; return ``None`` if it is unusable.

    Parameters
    ----------
    url:
        Raw URL string, possibly dirty.

    Returns
    -------
    Cleaned URL string or ``None`` if the URL should be discarded.
    """
    if not isinstance(url, str):
        return None
    url = url.strip().lower()
    if not url or len(url) > _MAX_URL_LEN:
        return None
    # Keep URLs that at least start with a scheme; drop obviously malformed
    # entries (e.g. bare file paths, empty strings, SQL fragments).
    if not url.startswith(("http://", "https://")):
        return None
    return url


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class UrlDataset(Dataset):
    """PyTorch Dataset wrapping tokenized URLs."""

    def __init__(
        self,
        urls: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            urls,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_raw_data(csv_path: str, url_col: str, label_col: str) -> pd.DataFrame:
    """Load and minimally validate the CSV dataset.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    url_col:
        Name of the URL column.
    label_col:
        Name of the integer label column.

    Returns
    -------
    DataFrame with exactly two columns: ``url`` and ``label``.

    Raises
    ------
    FileNotFoundError:
        If ``csv_path`` does not exist.
    ValueError:
        If required columns are missing or no valid rows remain after cleaning.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path, usecols=[url_col, label_col], dtype={url_col: str, label_col: object})

    if url_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{url_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns={url_col: "url", label_col: "label"})

    # Drop rows where label is not parseable as int
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Keep only binary labels
    df = df[df["label"].isin([0, 1])].copy()

    # Preprocess URLs
    df["url"] = df["url"].apply(preprocess_url)
    df = df.dropna(subset=["url"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    logger.info(
        "Loaded %d rows — benign: %d, malicious: %d",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )
    return df


def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Undersample the majority class so both classes have equal size.

    Parameters
    ----------
    df:
        DataFrame with ``url`` and ``label`` columns.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Balanced DataFrame.
    """
    majority_label = int(df["label"].value_counts().idxmax())
    minority_label = 1 - majority_label

    df_majority = df[df["label"] == majority_label]
    df_minority = df[df["label"] == minority_label]

    n_minority = len(df_minority)
    logger.info(
        "Balancing: undersampling majority class %d from %d → %d rows",
        majority_label,
        len(df_majority),
        n_minority,
    )

    df_majority_down = resample(
        df_majority, replace=False, n_samples=n_minority, random_state=seed
    )
    balanced = (
        pd.concat([df_majority_down, df_minority])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    logger.info("Balanced dataset size: %d rows", len(balanced))
    return balanced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    config_path: str = "config.yaml",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from a config YAML.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.

    Returns
    -------
    ``(train_loader, val_loader, test_loader)``
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_name = cfg["models"][0]  # tokenizer from first model in list

    seed = int(train_cfg["seed"])
    max_length = int(train_cfg["max_length"])
    batch_size = int(train_cfg["batch_size"])

    df = load_raw_data(data_cfg["path"], data_cfg["url_column"], data_cfg["label_column"])
    df = balance_dataset(df, seed=seed)

    urls = df["url"].tolist()
    labels = df["label"].tolist()

    val_size = float(data_cfg["val_split"])
    test_size = float(data_cfg["test_split"])
    # First split off test set, then split remainder into train+val
    relative_val = val_size / (1.0 - test_size)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        urls, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=y_train_val,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    train_ds = UrlDataset(X_train, y_train, tokenizer, max_length)
    val_ds = UrlDataset(X_val, y_val, tokenizer, max_length)
    test_ds = UrlDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
