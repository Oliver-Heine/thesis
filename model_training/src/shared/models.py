from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.backend_data_loader import SPECIAL_TOKENS
from shared.utils import logger


def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def build_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


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
