import argparse
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from shared.utils import logging, load_config
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple GPU (MPS)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def load_model(model_name, device):
    # Convert to absolute path if it's a local path
    if os.path.exists(model_name):
        model_path = os.path.abspath(model_name)
    else:
        model_path = model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return tokenizer, model


def predict_batch(texts, tokenizer, model, device):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs.pop("token_type_ids", None)

    # move tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    preds = torch.argmax(probs, dim=1).cpu().numpy()
    malicious_probs = probs[:, 1].cpu().numpy()

    return preds, malicious_probs

def save_summary_metrics(config):
    summary_file = f"../evaluation_results/{config['hf_train_version']}/metrics_summary.csv"
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Model", "Accuracy", "Precision", "Recall", "F1", "Specificity", "AUC", "Avg_Precision", "TP", "TN", "FP",
            "FN"
        ])
        for model_name in config["models"]:
            safe_name = model_name.replace("/", "_")
            # read each metrics file and append
            path = f"../evaluation_results/{config['hf_train_version']}/{safe_name}_metrics.txt"
            metrics = {}
            with open(path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, val = line.strip().split(":")
                        metrics[key.strip()] = val.strip()
            writer.writerow([
                safe_name,
                metrics.get("Accuracy", ""),
                metrics.get("Precision", ""),
                metrics.get("Recall", ""),
                metrics.get("F1 Score", ""),
                metrics.get("Specificity", ""),
                metrics.get("AUC", ""),
                metrics.get("Avg Precision", ""),
                metrics.get("TP", ""),
                metrics.get("TN", ""),
                metrics.get("FP", ""),
                metrics.get("FN", "")
            ])

def save_metrics(model_name, accuracy, precision, recall, f1, specificity,
                 auc, avg_precision, tp, tn, fp, fn, fold):

    safe_model_name = model_name.replace("/", "_")

    path = f"../evaluation_results/{safe_model_name}/fold_{fold}_metrics.txt"

    with open(path, "w") as f:

        f.write("Evaluation Results\n")
        f.write("-------------------\n")
        f.write(f"Accuracy:     {accuracy:.4f}\n")
        f.write(f"Precision:    {precision:.4f}\n")
        f.write(f"Recall:       {recall:.4f}\n")
        f.write(f"F1 Score:     {f1:.4f}\n")
        f.write(f"Specificity:  {specificity:.4f}\n")
        f.write(f"AUC:          {auc:.4f}\n")
        f.write(f"Avg Precision:{avg_precision:.4f}\n")

        f.write("\nConfusion Matrix\n")
        f.write(f"TP: {tp}\n")
        f.write(f"TN: {tn}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n")

def evaluate(model, tokenizer, dataset, device, model_name, fold):
    texts = dataset["url"]
    labels = np.array(dataset["result"])

    preds = []
    probs = []

    batch_size = 1024

    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch = texts[i:i+batch_size]

        batch_preds, batch_probs = predict_batch(batch, tokenizer, model, device)

        preds.extend(batch_preds)
        probs.extend(batch_probs)

    preds = np.array(preds)
    probs = np.array(probs)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    specificity = tn / (tn + fp)

    auc = roc_auc_score(labels, probs)
    avg_precision = average_precision_score(labels, probs)

    save_metrics(
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        auc,
        avg_precision,
        tp,
        tn,
        fp,
        fn,
        fold
    )

    return labels, preds, probs, accuracy, precision, recall, f1, specificity, auc, avg_precision, tp, tn, fp, fn

def plot_confusion(labels, preds, model_name, fold):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Benign","Malicious"],
        yticklabels=["Benign","Malicious"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # clean model name for filename
    safe_model_name = f"{model_name.replace('/', '_')}"

    path = f"../evaluation_results/{safe_model_name}/ConfusionMatrix/fold_{fold}_confusion_matrix.png"

    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_roc(labels, probs, model_name, fold):

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # clean model name for filename
    safe_model_name = f"{model_name.replace('/', '_')}"

    path = f"../evaluation_results/{safe_model_name}/ROC/fold_{fold}_roc.png"

    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_precision_recall(labels, probs, model_name, fold):

    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    plt.figure()

    plt.plot(recall, precision, label=f"AP = {ap:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # clean model name for filename
    safe_model_name = f"{model_name.replace('/', '_')}"

    path = f"../evaluation_results/{safe_model_name}/Precision-recall/fold_{fold}_Precision-Recall.png"

    plt.savefig(path, bbox_inches="tight")
    plt.close()

def main(config_path, fold):
    config = load_config(config_path)
    hf_train_version = config["hf_train_version"]

    os.makedirs(f"../evaluation_results/{hf_train_version}/ConfusionMatrix", exist_ok=True)
    os.makedirs(f"../evaluation_results/{hf_train_version}/ROC", exist_ok=True)
    os.makedirs(f"../evaluation_results/{hf_train_version}/Precision-recall", exist_ok=True)

    dataset = load_from_disk("data/splits")

    test_dataset = dataset["test"]

    device = get_device()

    for model_name in config["models"]:

        hf_model = (
            config["hf_username"]
            + model_name
            + hf_train_version
        )

        logging.info("\n================================")
        logging.info(f"Evaluating: {hf_model}")

        tokenizer, model = load_model(hf_model, device)

        labels, preds, probs = evaluate(model, tokenizer, test_dataset, device, model_name, hf_train_version)

        plot_confusion(labels, preds, model_name, hf_train_version, fold)
        plot_roc(labels, probs, model_name, hf_train_version, fold)
        plot_precision_recall(labels, probs, model_name, hf_train_version, fold)

    save_summary_metrics(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")

    args = parser.parse_args()

    main(args.config)