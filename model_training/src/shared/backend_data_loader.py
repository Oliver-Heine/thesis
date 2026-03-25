import argparse
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from utils import logger
import pandas as pd
import os
from utils import load_config


# ---------------------------
# Feature Engineering (TEXT)
# ---------------------------

def bucket_page_size(value):
    if value < 50:
        return "<PAGE_SIZE_SMALL>"
    elif value < 150:
        return "<PAGE_SIZE_MEDIUM>"
    elif value < 300:
        return "<PAGE_SIZE_LARGE>"
    else:
        return "<PAGE_SIZE_XL>"


def bucket_tld_entropy(value):
    if value < 1.2:
        return "<TLD_ENTROPY_LOW>"
    elif value < 1.8:
        return "<TLD_ENTROPY_MEDIUM>"
    else:
        return "<TLD_ENTROPY_HIGH>"


def bucket_count(value, name):
    if value == 0:
        return f"<{name}_0>"
    elif value == 1:
        return f"<{name}_1>"
    elif value <= 3:
        return f"<{name}_2_3>"
    elif value <= 7:
        return f"<{name}_4_7>"
    elif value <= 15:
        return f"<{name}_8_15>"
    else:
        return f"<{name}_16_PLUS>"


def row_to_text(row):
    tokens = []

    # Domain (raw)
    tokens.append(f"<DOMAIN> {row['domain']}")

    # Count-based features
    tokens.append(bucket_count(int(row["redirect_count"]), "REDIRECT_COUNT"))
    tokens.append(bucket_count(int(row["third_party_domains"]), "THIRD_PARTY_DOMAINS"))
    tokens.append(bucket_count(int(row["num_inputs"]), "NUM_INPUTS"))
    tokens.append(bucket_count(int(row["num_iframes"]), "NUM_IFRAMES"))
    tokens.append(bucket_count(int(row["external_scripts"]), "EXTERNAL_SCRIPTS"))

    # Boolean features
    tokens.append("<LOGIN_FORM_YES>" if row["login_form"] == 1 else "<LOGIN_FORM_NO>")
    tokens.append("<PASSWORD_INPUT_YES>" if row["password_input"] == 1 else "<PASSWORD_INPUT_NO>")
    tokens.append("<USES_EVAL_YES>" if row["uses_eval"] == 1 else "<USES_EVAL_NO>")
    tokens.append("<POPUP_YES>" if row["popup_window"] == 1 else "<POPUP_NO>")
    tokens.append("<DOC_LOC_CHANGE_YES>" if row["document_location_change"] == 1 else "<DOC_LOC_CHANGE_NO>")
    tokens.append("<CANVAS_FP_YES>" if row["canvas_fingerprint"] == 1 else "<CANVAS_FP_NO>")

    # Continuous features (bucketed)
    tokens.append(bucket_page_size(float(row["page_size"])))
    tokens.append(bucket_tld_entropy(float(row["tld_entropy"])))

    # Cert
    tokens.append("<CERT_VALID_YES>" if row["cert_valid"] == 1 else "<CERT_VALID_NO>")

    # Domain age
    if int(row["domain_age"]) == -1:
        tokens.append("<DOMAIN_AGE_UNKNOWN>")
    else:
        tokens.append(bucket_count(int(row["domain_age"]), "DOMAIN_AGE"))

    return " ".join(tokens)


# ---------------------------
# Dataset Loading
# ---------------------------

def load_dataset_from_config(dataset_config):

    if os.path.exists("data/backend_splits"):
        logger.info("Loading existing backend dataset splits...")
        return load_from_disk("data/backend_splits")

    logger.info(f"Loading dataset: {dataset_config['path']}")

    dataframe = pd.read_csv(dataset_config["path"])

    # Convert to text representation
    logger.info("Transforming rows into text representation...")
    dataframe["text"] = dataframe.apply(row_to_text, axis=1)

    dataframe = dataframe[["text", "label"]]

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
# CLI (optional utility)
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    load_dataset_from_config(config["dataset_backend"])