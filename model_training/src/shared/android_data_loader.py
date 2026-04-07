import argparse
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from utils import logger
import pandas as pd
import os
import re
import tldextract
from urllib.parse import urlparse
from utils import load_config


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

def load_dataset_from_config(dataset_config):

    if os.path.exists("../data/splits"):
        logger.info("Loading existing dataset splits...")
        return load_from_disk("../data/splits")

    logger.info(f"Loading dataset: {dataset_config['path']}")

    dataframe = pd.read_csv(dataset_config["path"])

    dataset = Dataset.from_pandas(dataframe)

    dataset = dataset.cast_column("result", ClassLabel(num_classes=2, names=["0", "1"]))

    logger.info(f"Splitting dataset into train, test, and validation...")
    train_size = dataset_config["train_split"]
    temp_size = 1 - train_size

    train_test = dataset.train_test_split(
        test_size=temp_size,
        seed=42,
        stratify_by_column="result"
    )

    train_dataset = train_test["train"]
    temp_dataset = train_test["test"]

    val_ratio = dataset_config["val_split"] / (dataset_config["val_split"] + dataset_config["test_split"])

    val_test = temp_dataset.train_test_split(
        test_size=1 - val_ratio,
        seed=42,
        stratify_by_column="result"
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

    dataset_dict.save_to_disk("data/splits")

    return dataset_dict

def tokenize_dataset(dataset, tokenizer):

    def _tokenize(batch):
        return tokenizer(
            batch["url"],
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(_tokenize, batched=True)

    tokenized_dataset = tokenized_dataset.rename_column("result", "labels")

    tokenized_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset

def normalize_and_remove_duplicates(dataset_config):
    logger.info(f"Loading dataset: {dataset_config['path']}")

    dataframe = pd.read_csv(dataset_config["path"])

    dataframe["normalized"] = dataframe["url"].apply(normalize_url)
    dataframe = dataframe[dataframe["normalized"].notna()]

    dataframe = dataframe.drop_duplicates(subset="normalized", keep="last")

    dataframe = dataframe[["normalized", "result"]]
    dataframe = dataframe.rename(columns={"normalized": "url"})
    dataframe.to_csv("../combined_urls_no_duplicated.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    normalize_and_remove_duplicates(config['dataset'])