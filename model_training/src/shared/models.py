from transformers import AutoModelForSequenceClassification, AutoTokenizer

def build_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def build_model(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


def build_tokenizer_backend(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    custom_tokens = generate_all_tokens()

    tokenizer.add_tokens(custom_tokens)

    return tokenizer

def build_model_backend(model_name, num_labels, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels)

    model.resize_token_embeddings(len(tokenizer))

    return model

def generate_all_tokens():
    tokens = []

    # Boolean
    tokens.extend([
        "<DOMAIN>", "<LOGIN_FORM_YES>", "<LOGIN_FORM_NO>",
        "<PASSWORD_INPUT_YES>", "<PASSWORD_INPUT_NO>",
        "<USES_EVAL_YES>", "<USES_EVAL_NO>",
        "<POPUP_YES>", "<POPUP_NO>",
        "<DOC_LOC_CHANGE_YES>", "<DOC_LOC_CHANGE_NO>",
        "<CANVAS_FP_YES>", "<CANVAS_FP_NO>",
        "<CERT_VALID_YES>", "<CERT_VALID_NO>",
        "<DOMAIN_AGE_UNKNOWN>",
    ])

    # Continuous
    tokens.extend([
        "<PAGE_SIZE_SMALL>", "<PAGE_SIZE_MEDIUM>", "<PAGE_SIZE_LARGE>", "<PAGE_SIZE_XL>",
        "<TLD_ENTROPY_LOW>", "<TLD_ENTROPY_MEDIUM>", "<TLD_ENTROPY_HIGH>",
    ])

    # Count buckets
    features = [
        "REDIRECT_COUNT",
        "THIRD_PARTY_DOMAINS",
        "NUM_INPUTS",
        "NUM_IFRAMES",
        "EXTERNAL_SCRIPTS",
        "DOMAIN_AGE"
    ]

    buckets = ["0", "1", "2_3", "4_7", "8_15", "16_PLUS"]

    for feature in features:
        for bucket in buckets:
            tokens.append(f"<{feature}_{bucket}>")

    return tokens
