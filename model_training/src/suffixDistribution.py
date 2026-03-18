import argparse
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from utils import logger
import pandas as pd
import os
import re
import tldextract
from urllib.parse import urlparse
from utils import load_config


def extract_suffix(url: str):

    url = url.lower().strip()

    # remove protocol and www for normalization
    url_clean = re.sub(r"^https?://", "", url)
    url_clean = re.sub(r"^www\.", "", url_clean)

    # Ensure urlparse sees a scheme, otherwise domain may be misparsed
    parsed = urlparse("http://" + url_clean)
    ext = tldextract.extract(url_clean)

    if ext.suffix == "":
        return "ipaddress"

    return ext.suffix


def count_suffixes_and_remove_duplicates(dataset_config):
    logger.info(f"Loading dataset: {dataset_config['path']}")

    dataframe = pd.read_csv(dataset_config["path"])
    # Normalize header names to handle inputs like "id, url".
    dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]

    if "url" not in dataframe.columns:
        raise ValueError(
            f"Expected a 'url' column in {dataset_config['path']}. "
            f"Found columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe[dataframe["url"].notna()]
    dataframe["url"] = dataframe["url"].astype(str).str.strip()
    dataframe = dataframe[dataframe["url"] != ""]
    dataframe = dataframe.drop_duplicates(subset="url", keep="last")
    dataframe["suffix"] = dataframe["url"].apply(extract_suffix)
    dataframe = dataframe[dataframe["suffix"].notna()]

    suffix_counts = dataframe["suffix"].value_counts().reset_index()
    suffix_counts.columns = ["suffix", "count"]
    dataframe = suffix_counts

    total_count = dataframe["count"].sum()
    logger.info(f"Total suffix count: {total_count}")

    dataframe = dataframe[["suffix", "count"]]
    dataframe = dataframe.rename(columns={"suffix": "url"})
    dataframe.to_csv("../combined_urls_test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    count_suffixes_and_remove_duplicates(config['dataset'])