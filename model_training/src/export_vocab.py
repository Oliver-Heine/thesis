import argparse
from pathlib import Path

from transformers import AutoTokenizer

from utils import load_config, logger


URL_TAG_TOKENS = ["<subdomain>", "<domain>", "<suffix>", "<path>", "<query>"]


def build_model_id(config: dict, model_name: str) -> str:
    username = config["hf_username"].rstrip("/")
    return f"{username}/{model_name}{config['hf_train_version']}"


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_model_candidates(config: dict, model_name: str) -> list[str]:
    username = config["hf_username"].rstrip("/")
    train_version = config["hf_train_version"]
    safe_name = safe_model_name(model_name)
    this_dir = Path(__file__).resolve().parent

    return [
        str(this_dir / "output" / f"{safe_name}{train_version}"),
        str(Path.cwd() / "output" / f"{safe_name}{train_version}"),
        f"{username}/{safe_name}{train_version}",
        f"{username}/{model_name}{train_version}",
    ]


def load_tokenizer_with_fallback(config: dict, model_name: str):
    candidates = build_model_candidates(config, model_name)
    last_error = None

    for candidate in candidates:
        try:
            logger.info("Trying tokenizer source: %s", candidate)
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            logger.info("Resolved tokenizer source: %s", candidate)
            return tokenizer, candidate
        except Exception as exc:
            last_error = exc
            logger.warning("Failed tokenizer source: %s (%s)", candidate, exc)

    raise RuntimeError(
        f"Could not load tokenizer for '{model_name}'. Tried: {candidates}"
    ) from last_error


def ensure_url_tag_tokens(tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in URL_TAG_TOKENS if tok not in vocab]

    if not missing:
        logger.info("All URL tag tokens already present in tokenizer vocab.")
        return 0

    logger.warning("Missing URL tag tokens: %s", missing)

    # For BERT/DistilBERT WordPiece, these should behave as indivisible tokens.
    added = tokenizer.add_tokens(missing, special_tokens=False)
    logger.info("Added %d missing URL tag tokens.", added)
    return added


def validate_required_tokens(tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in URL_TAG_TOKENS if tok not in vocab]
    if missing:
        raise RuntimeError(
            f"Tokenizer still missing required URL tag tokens after patching: {missing}"
        )

    for tok in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        if tok not in vocab:
            raise RuntimeError(f"Tokenizer missing required base token: {tok}")


def write_vocab_txt(tokenizer, output_path: Path) -> int:
    vocab = tokenizer.get_vocab()
    ordered_tokens = [
        token for token, _ in sorted(vocab.items(), key=lambda item: item[1])
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for token in ordered_tokens:
            f.write(token + "\n")

    return len(ordered_tokens)


def export_model_vocab(config: dict, model_name: str, output_root: Path) -> None:
    tokenizer, resolved_source = load_tokenizer_with_fallback(config, model_name)

    # Patch missing custom URL structure tokens before export.
    ensure_url_tag_tokens(tokenizer)
    validate_required_tokens(tokenizer)

    model_output_dir = output_root / safe_model_name(model_name)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save patched tokenizer assets.
    tokenizer.save_pretrained(model_output_dir)

    vocab_path = model_output_dir / "vocab.txt"
    vocab_size = write_vocab_txt(tokenizer, vocab_path)

    logger.info(
        "Saved %s tokens to %s (source: %s)",
        vocab_size,
        vocab_path,
        resolved_source,
    )


def main(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    models = config.get("models", [])

    if not models:
        raise ValueError("No models found in config under 'models'.")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        export_model_vocab(config, model_name, output_root)

    logger.info("Finished exporting vocab files to %s", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export vocab.txt files from trained Hugging Face tokenizers."
    )
    parser.add_argument("--config", default="../config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--output-dir",
        default="../model_vocabs",
        help="Directory where per-model vocab.txt files will be written.",
    )

    args = parser.parse_args()
    main(args.config, args.output_dir)