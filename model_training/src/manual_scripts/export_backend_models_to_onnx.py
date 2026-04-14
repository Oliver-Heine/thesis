#!/usr/bin/env python3
"""
Convert backend Hugging Face models to ONNX format for use in Java applications.
Supports dynamic batch size and sequence length.
"""

import argparse
import inspect
import logging
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_backend_models_to_onnx")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_input_path(path_value: str) -> Path:
    """Resolve an input path from cwd first, then project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


def safe_model_name(model_name: str) -> str:
    """Convert model name to safe filename."""
    return model_name.replace("/", "_")


def build_backend_repo_id(model_name: str, hf_username: str, train_version: str) -> str:
    """Build trained backend repo id using config naming convention."""
    repo_name = safe_model_name(model_name)
    return f"{hf_username}/{repo_name}{train_version}"


def build_dummy_inputs(model, tokenizer=None, seq_len: int = 32) -> dict[str, torch.Tensor]:
    """Create export inputs, with tokenizer fallback for broken tokenizer configs."""
    if tokenizer is not None:
        encoded = tokenizer("https://example.com", return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"]
        return inputs

    vocab_size = int(getattr(model.config, "vocab_size", 30522) or 30522)
    safe_vocab_size = max(vocab_size, 2)
    input_ids = torch.randint(0, safe_vocab_size, (1, seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def export_model_to_onnx(
    model_name: str,
    output_path: Path,
    hf_username: str,
    train_version: str,
    hf_token: str | None,
) -> bool:
    """
    Export a Hugging Face model to ONNX format.
    
    Args:
        model_name: Model name or HuggingFace repo ID
        output_path: Path to save ONNX file
        hf_username: Optional HuggingFace username prefix for custom models
    
    Returns:
        True if successful, False otherwise
    """
    try:
        trained_model_id = build_backend_repo_id(model_name, hf_username, train_version)
        token_kwargs = {"token": hf_token} if hf_token else {}

        logger.info("Loading model: %s", trained_model_id)

        # --------- Load model ---------
        model = AutoModelForSequenceClassification.from_pretrained(trained_model_id, **token_kwargs)
        model.eval()

        # --------- Load tokenizer (fallback-safe) ---------
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(trained_model_id, **token_kwargs)
        except Exception as exc:
            logger.warning(
                "Tokenizer load failed for %s (%s). Falling back to model-config dummy inputs.",
                trained_model_id,
                exc,
            )

        logger.info("Model loaded successfully: %s", trained_model_id)

        # --------- Prepare dummy input ---------
        inputs = build_dummy_inputs(model, tokenizer=tokenizer)
        input_names = ["input_ids", "attention_mask"]
        export_tensors = (inputs["input_ids"], inputs["attention_mask"])

        signature = inspect.signature(model.forward)
        supports_token_type_ids = "token_type_ids" in signature.parameters
        if supports_token_type_ids:
            token_type_ids = inputs.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(inputs["input_ids"])
            input_names.append("token_type_ids")
            export_tensors = (inputs["input_ids"], inputs["attention_mask"], token_type_ids)

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        }
        if supports_token_type_ids:
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "seq_len"}
        
        # --------- Export to ONNX ---------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting to ONNX: %s", output_path)
        torch.onnx.export(
            model,
            export_tensors,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )
        logger.info("ONNX model exported to: %s", output_path)
        
        # --------- Validate ONNX model ---------
        logger.info("Validating ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed!")
        
        # --------- Test ONNX inference ---------
        logger.info("Testing ONNX inference...")
        ort_session = ort.InferenceSession(str(output_path))
        ort_feed = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        if supports_token_type_ids:
            ort_feed["token_type_ids"] = export_tensors[2].numpy()

        outputs = ort_session.run(None, ort_feed)
        logger.info("ONNX inference successful! Output shape: %s", outputs[0].shape)
        logger.info("Sample logits: %s", outputs[0][:5])
        
        return True
        
    except Exception as exc:
        logger.error("Failed to export model '%s': %s", model_name, exc)
        return False


def main(config_path: str, output_dir: str, hf_username: str = None) -> None:
    """
    Export all backend models from config to ONNX format.
    
    Args:
        config_path: Path to config.yaml
        output_dir: Directory to save ONNX files
        hf_username: Optional HuggingFace username for custom models
    """
    resolved_config_path = resolve_input_path(config_path)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    config = load_config(str(resolved_config_path))
    models = config.get("models_backend", [])
    
    if not models:
        raise ValueError("No models found in config under 'models_backend'.")
    
    if hf_username is None:
        hf_username = config.get("hf_username", "").rstrip("/")
    train_version = config.get("hf_backend_train_version", "")
    hf_token = config.get("hf_token")

    if not hf_username:
        raise ValueError("hf_username is missing in config and not provided via --hf-username.")
    if not train_version:
        raise ValueError("hf_backend_train_version is missing in config.")
    
    output_root = resolve_input_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    logger.info("Exporting %d backend models to ONNX format", len(models))
    logger.info("Output directory: %s", output_root)
    
    successful = []
    failed = []
    
    for model_name in models:
        safe_name = safe_model_name(model_name)
        output_path = output_root / f"{safe_name}.onnx"
        
        if export_model_to_onnx(model_name, output_path, hf_username, train_version, hf_token):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Export Summary:")
    logger.info("Successful: %d/%d", len(successful), len(models))
    if successful:
        for model in successful:
            logger.info("  ✓ %s", model)
    
    if failed:
        logger.warning("Failed: %d/%d", len(failed), len(models))
        for model in failed:
            logger.warning("  ✗ %s", model)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export backend Hugging Face models to ONNX format for Java applications."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_onnx_backend",
        help="Directory to save ONNX files"
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default=None,
        help="HuggingFace username (optional, overrides config)"
    )
    
    args = parser.parse_args()
    main(args.config, args.output_dir, args.hf_username)
