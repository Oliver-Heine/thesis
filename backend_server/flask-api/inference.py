import argparse

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_config, logger


def load_model(config_path="./config.yaml"):
    config = load_config(config_path)
    logger.info(f"Logging into huggingface")
    login(token=config["hf_token"])
    model_name = config["hf_username"] + config["models_backend"][2] + config["hf_backend_train_version"]  # or local path

    logger.info(f"Loading model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    logger.info(f"Model loaded.")
    return tokenizer, model


def predict(predict_input, tokenizer, model):
    inputs = tokenizer(predict_input, return_tensors="pt", truncation=True, max_length=256)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return pred, confidence