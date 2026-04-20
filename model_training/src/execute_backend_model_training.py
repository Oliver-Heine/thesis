import argparse
from huggingface_hub import login

from shared.utils import load_config, logger
from shared.backend_data_loader import load_raw_dataset, tokenize_dataset
from shared.k_fold_generator import generate_folds
from shared.models import build_tokenizer_backend, build_model_backend
from shared.train import train
from datasets import DatasetDict
import json
import os
from execute_android_evaluate import (
    evaluate,
    plot_confusion,
    plot_roc,
    plot_precision_recall,
    get_device
)

CHECKPOINT_FILE = "training_backend_checkpoint.json"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

def append_result(result):
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

def main(config_path: str):

    config = load_config(config_path)

    login(token=config["hf_token"])

    dataframe = load_raw_dataset(config["dataset_backend"])
    checkpoint = load_checkpoint()

    device = get_device()

    for model_name in config["models_backend"]:
        model_key = model_name.replace("/", "_")

        os.makedirs(f"../evaluation_results/{model_key}/ConfusionMatrix", exist_ok=True)
        os.makedirs(f"../evaluation_results/{model_key}/ROC", exist_ok=True)
        os.makedirs(f"../evaluation_results/{model_key}/Precision-recall", exist_ok=True)


        tokenizer = build_tokenizer_backend(model_name)
        for fold_data in generate_folds(dataframe, n_splits=10):

            fold = fold_data["fold"]

            if fold in checkpoint.get(model_key, {}).get("completed_folds", []):
                continue

            logger.info(f"Training {model_key} - Fold {fold}")

            dataset_dict = DatasetDict({
                "train": fold_data["train"],
                "validation": fold_data["validation"],
                "test": fold_data["test"]
            })

            tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer, config["training_backend"]["max_length"])

            model = build_model_backend(model_name, num_labels=2, tokenizer=tokenizer)

            output_dir = f"output/{model_key}/fold_{fold}"

            trainer = train(
                model,
                tokenized_dataset,
                tokenizer,
                config["training_backend"],
                f"{model_key}_fold_{fold}",
                output_dir=output_dir,
            )

            trained_model = trainer.model
            trained_model.eval()

            labels, preds, probs, accuracy, precision, recall, f1, specificity, auc, avg_precision, tp, tn, fp, fn = (
                evaluate(
                    trained_model,
                    tokenizer,
                    fold_data["test"],  # 👈 raw dataset (NOT tokenized!)
                    device,
                    model_key,
                    fold
                ))

            plot_confusion(labels, preds, model_key, fold)
            plot_roc(labels, probs, model_key, fold)
            plot_precision_recall(labels, probs, model_key, fold)

            append_result({
                "model": model_key,
                "fold": fold,
                "accuracy": float(accuracy),
                "specificity": float(specificity),
                "precision": float(precision),
                "avg_precision": float(avg_precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            })

            checkpoint.setdefault(model_key, {}).setdefault("completed_folds", []).append(fold)
            save_checkpoint(checkpoint)
            trainer.push_to_hub(commit_message=f"{model_key} fold {fold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    main(args.config)