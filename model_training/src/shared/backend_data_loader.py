import argparse
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from shared.utils import logger
import pandas as pd
import numpy as np
import os
from shared.utils import load_config
import re
from urllib.parse import urlparse
import tldextract


# ---------------------------
# Feature Definitions
# ---------------------------

SPECIAL_TOKENS = ["<DOMAIN>", "<SUBDOMAIN>", "<PATH>", "<QUERY>", "<SUFFIX>"]

NUMERICAL_FEATURES = [
    "redirect_count",
    "server_redirect_count",
    "third_party_domains",
    "num_inputs",
    "num_iframes",
    "external_scripts",
    "page_size",
]

NUMERICAL_BINS = {
    "redirect_count": [0, 1, 2, 3, 4, 5, 6, 7, 8, "MAX"],
    "server_redirect_count": [0, 1, 2, 3, 4, 5, 6, 7, 8, "MAX"],
    "third_party_domains": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, "MAX"],
    "num_inputs": [0, 1, 2, 3, 4, 5, 10, 15, 20, "MAX"],
    "num_iframes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, "MAX"],
    "external_scripts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, "MAX"],
    "page_size": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, "MAX"]
}

BOOLEAN_FEATURES = [
    "login_form",
    "password_input",
    "uses_eval",
    "canvas_fingerprint",
    "cert_valid"
]

def analyze_numeric_features(df, numeric_features=NUMERICAL_FEATURES, output_csv="data/numeric_distribution.txt"):
    """
    Prints and optionally saves the distribution of numeric features.

    Args:
        df (pd.DataFrame): The dataset dataframe.
        numeric_features (list[str]): List of numeric column names to analyze.
        output_csv (str, optional): Path to save the distributions as CSV.
    """
    distributions = {}

    for feature in numeric_features:
        counts = df[feature].value_counts(dropna=False).sort_index()
        distributions[feature] = counts
        print(f"\nFeature: {feature}")
        print(counts)

    if output_csv:
        # Combine all distributions into one CSV
        with pd.ExcelWriter(output_csv) if output_csv.endswith(".xlsx") else open(output_csv, "w") as f:
            for feature, counts in distributions.items():
                f.write(f"Feature: {feature}\n")
                counts.to_csv(f, header=["count"])
                f.write("\n")
        print(f"\nSaved numeric distributions to {output_csv}")

def normalize_url(url: str):
    url = url.lower().strip()

    # remove protocol and www for normalization
    url_clean = re.sub(r"^https?://", "", url)
    url_clean = re.sub(r"^www\.", "", url_clean)

    # Ensure urlparse sees a scheme, otherwise domain may be misparsed
    parsed = urlparse("http://" + url_clean)
    ext = tldextract.extract(url_clean)

    tokens = []

    # subdomain
    if ext.subdomain:
        tokens.append("<subdomain>")
        tokens.extend(ext.subdomain.split("."))

    # domain
    tokens.append("<domain>")
    tokens.append(ext.domain)

    # domain extension
    tokens.append("<suffix>")
    tokens.extend(ext.suffix.split("."))

    # path
    if parsed.path and parsed.path != "/":
        tokens.append("<path>")
        tokens.extend(re.split(r"[/\-_.?=&]", parsed.path))

    # query
    if parsed.query:
        tokens.append("<query>")
        tokens.extend(re.split(r"[=&]", parsed.query))

    return " ".join([t for t in tokens if t])

# ---------------------------
# Vectorized Transformation
# ---------------------------

def prepare_bins(df, template=NUMERICAL_BINS, special_tokens=SPECIAL_TOKENS):
    """
    Returns the numeric bins for each feature.
    Optionally, extends special_tokens with <FEATURE_BUCKET> tokens for all buckets.
    """
    bins = {}
    if special_tokens is None:
        special_tokens = []

    for feature, edges in template.items():
        new_edges = []
        for e in edges:
            if e == "MAX":
                max_val = df[feature].max()
                new_edges.append(max_val + 1)
            else:
                new_edges.append(e)
        bins[feature] = new_edges

        # Add tokens for all buckets
        n_buckets = len(new_edges) - 1
        for i in range(n_buckets):
            special_tokens.append(f"<{feature.upper()}_{i}>")
        special_tokens.append(f"<{feature.upper()}_MISSING>")  # missing value token

    return bins, special_tokens


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---------------------------
    # Prepare numeric bins and extend SPECIAL_TOKENS
    # ---------------------------
    bins_formatted, SPECIAL_TOKENS = prepare_bins(df)

    # ---------------------------
    # Numerical features
    # ---------------------------
    for feature in NUMERICAL_FEATURES:
        values = df[feature].astype(float)
        mask_nan = values.isna()
        non_nan_values = values[~mask_nan]

        bucket_indices = pd.cut(
            non_nan_values,
            bins=bins_formatted[feature],
            labels=False,
            include_lowest=True
        )

        tokens = [f"<{feature.upper()}_{b}>" for b in bucket_indices]

        full_tokens = []
        idx = 0
        for is_nan in mask_nan:
            if is_nan:
                full_tokens.append(f"<{feature.upper()}_MISSING>")
            else:
                full_tokens.append(tokens[idx])
                idx += 1

        df[feature + "_token"] = full_tokens

    # ---------------------------
    # Boolean features
    # ---------------------------
    for feature in BOOLEAN_FEATURES:
        yes_token = f"<{feature.upper()}_YES>"
        no_token = f"<{feature.upper()}_NO>"

        # Add boolean tokens only if not already present
        if yes_token not in SPECIAL_TOKENS:
            SPECIAL_TOKENS.append(yes_token)
        if no_token not in SPECIAL_TOKENS:
            SPECIAL_TOKENS.append(no_token)

        df[feature + "_token"] = np.where(
            df[feature] == 1,
            yes_token,
            no_token
        )

    # ---------------------------
    # Combine all tokens into a single text column
    # ---------------------------
    token_columns = [f"{f}_token" for f in NUMERICAL_FEATURES + BOOLEAN_FEATURES]
    df["text"] = df["domain"].apply(normalize_url) + " " + df[token_columns].agg(" ".join, axis=1)

    return df[["text", "label"]]


# ---------------------------
# Dataset Loading
# ---------------------------

def load_dataset_from_config(dataset_config):

    if os.path.exists("data/backend_splits"):
        logger.info("Loading existing backend dataset splits...")
        return load_from_disk("data/backend_splits")

    logger.info(f"Loading dataset: {dataset_config['path']}")

    dataframe = pd.read_csv(dataset_config["path"])

    # Transform dataset (FAST)
    logger.info("Transforming dataset (vectorized)...")
    dataframe = transform_dataframe(dataframe)
    logger.info("Transformation complete.")

    dataset = Dataset.from_pandas(dataframe)

    dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=["0", "1"]))

    # ---------------------------
    # Splitting
    # ---------------------------

    logger.info("Splitting dataset into train, validation, and test...")

    train_size = dataset_config["train_split"]
    temp_size = 1 - train_size

    train_test = dataset.train_test_split(
        test_size=temp_size,
        seed=42,
        stratify_by_column="label"
    )

    train_dataset = train_test["train"]
    temp_dataset = train_test["test"]

    val_ratio = dataset_config["val_split"] / (
        dataset_config["val_split"] + dataset_config["test_split"]
    )

    val_test = temp_dataset.train_test_split(
        test_size=1 - val_ratio,
        seed=42,
        stratify_by_column="label"
    )

    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    logger.info("Dataset sizes:")
    logger.info("Train: %d", len(train_dataset))
    logger.info("Validation: %d", len(val_dataset))
    logger.info("Test: %d", len(test_dataset))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    dataset_dict.save_to_disk("data/backend_splits")

    return dataset_dict


# ---------------------------
# Tokenization
# ---------------------------

def tokenize_dataset(dataset, tokenizer, max_length):

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text"]
    )

    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    tokenized_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    load_dataset_from_config(config["dataset_backend"])