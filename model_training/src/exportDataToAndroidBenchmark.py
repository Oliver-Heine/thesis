import argparse
from utils import load_config
from data import normalize_url
import pandas as pd

def main(config_path: str, output_path="benchmark_dataset.csv"):
    from datasets import Dataset, ClassLabel

    config = load_config(config_path)

    dataset_config = config["dataset"]
    dataframe = remove_duplicates_without_normalizing(dataset_config)

    dataset = Dataset.from_pandas(dataframe)

    dataset = dataset.cast_column(
        "result",
        ClassLabel(num_classes=2, names=["0", "1"])
    )

    # 🔹 SAME splitting logic (seed MUST match)
    train_size = dataset_config["train_split"]
    temp_size = 1 - train_size

    train_test = dataset.train_test_split(
        test_size=temp_size,
        seed=42,
        stratify_by_column="result"
    )

    temp_dataset = train_test["test"]

    val_ratio = dataset_config["val_split"] / (
        dataset_config["val_split"] + dataset_config["test_split"]
    )

    val_test = temp_dataset.train_test_split(
        test_size=1 - val_ratio,
        seed=42,
        stratify_by_column="result"
    )

    test_dataset = val_test["test"]

    # 🔹 Keep only required columns
    test_dataset = test_dataset.select_columns(["url", "result"])

    # 🔹 Convert label to int (VERY IMPORTANT for Android)
    test_dataset = test_dataset.map(lambda x: {"result": int(x["result"])})

    # 🔹 Export to CSV
    test_dataset.to_csv(output_path, index=False)

    print(f"✅ Test dataset exported to {output_path}")
    print(f"Samples: {len(test_dataset)}")

def remove_duplicates_without_normalizing(dataset_config):
    dataframe = pd.read_csv(dataset_config["path"])

    # 🔹 Create normalized column ONLY for deduplication
    dataframe["normalized"] = dataframe["url"].apply(normalize_url)

    # Remove rows where normalization failed
    dataframe = dataframe[dataframe["normalized"].notna()]

    # 🔹 Drop duplicates based on normalized value
    dataframe = dataframe.drop_duplicates(subset="normalized", keep="last")

    # 🔹 KEEP ORIGINAL URL (do NOT overwrite it)
    return dataframe[["url", "result"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    main(args.config)