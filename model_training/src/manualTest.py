import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_config
from data import normalize_url


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict(url, tokenizer, model):
    inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return pred, confidence

def main(config_path: str):
    config = load_config(config_path)
    model_name = config["hf_username"] + config["models"][1] + config["hf_train_version"] # or local path

    tokenizer, model = load_model(model_name)

    print("Model loaded. Enter URLs (type 'exit' to quit)\n")

    while True:
        url = input("URL> ")

        if url.lower() == "exit":
            break

        url = normalize_url(url)

        pred, conf = predict(url, tokenizer, model)

        label = "MALICIOUS" if pred == 1 else "BENIGN"

        print(f"Prediction: {label} (confidence {conf:.3f})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    args = parser.parse_args()

    main(args.config)