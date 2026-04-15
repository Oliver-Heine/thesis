from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.backend_data_loader import SPECIAL_TOKENS
from shared.utils import logger


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    SPECIAL_TOKENS = ["<domain>", "<subdomain>", "<path>", "<query>", "<suffix>"]
    tokenizer.add_tokens(SPECIAL_TOKENS)
    return tokenizer

def build_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

def load_pretrained_model_from_disk(model_key, fold, device):
    trained_model_path = f"output/{model_key}/fold_{fold}"

    trained_model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)
    trained_model.to(device)
    trained_model.eval()
    return trained_model


def build_tokenizer_backend(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"SPECIAL TOKENS to add: {SPECIAL_TOKENS}")
    tokenizer.add_tokens(SPECIAL_TOKENS)

    return tokenizer

def build_model_backend(model_name, num_labels, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels)

    model.resize_token_embeddings(len(tokenizer))

    return model
