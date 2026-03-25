import tensorflow as tf
from transformers import AutoTokenizer
import os
import yaml
import shutil
import argparse
from typing import Optional

# --- Helper function to load TF model with fallbacks ---
def load_tf_model(model_path: str, hf_token: Optional[str]):
    """Load TensorFlow sequence classification model from local or HF checkpoint."""
    try:
        from transformers import TFAutoModelForSequenceClassification
    except Exception as exc:
        raise RuntimeError(
            "Your installed transformers version has no TensorFlow classes. "
            "Install a TF-compatible version, for example: pip install \"transformers<5\""
        ) from exc

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    # Local training outputs are typically PyTorch checkpoints (model.safetensors).
    # from_pt=True lets TF classes load from those weights.
    if os.path.isdir(model_path):
        has_pt_weights = (
            os.path.exists(os.path.join(model_path, "model.safetensors"))
            or os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
        )
        has_tf_weights = os.path.exists(os.path.join(model_path, "tf_model.h5"))
        if has_pt_weights and not has_tf_weights:
            kwargs["from_pt"] = True
    else:
        kwargs["from_pt"] = True

    return TFAutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)

# --- Resolve paths relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert trained models to TFLite + vocab.txt")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to process (original name or slash-replaced name).",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable TFLite optimization/quantization (can help problematic conversions).",
    )
    return parser.parse_args()


args = parse_args()

# --- Load config ---
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

models_to_convert = config["models"]
hf_username = config["hf_username"]
hf_train_version = config["hf_train_version"]
hf_token = config.get("hf_token")

if args.models:
    requested = set(args.models)
    models_to_convert = [
        name
        for name in models_to_convert
        if name in requested or name.replace("/", "_") in requested
    ]

# --- Output directory ---
output_base_dir = os.path.join(SCRIPT_DIR, "tflite")
os.makedirs(output_base_dir, exist_ok=True)

# --- Helper function to get local model path ---
def get_local_model_path(model_name):
    """Convert model name to local directory format."""
    # Replace "/" with "_" for local path
    local_model_name = model_name.replace("/", "_")
    local_path = os.path.join(SCRIPT_DIR, "output", f"{local_model_name}{hf_train_version}")
    return local_path if os.path.exists(local_path) else None

# --- Helper function to get HuggingFace model path ---
def get_hf_model_path(model_name):
    """Convert model name to HuggingFace format."""
    base = hf_username.rstrip("/")
    return f"{base}/{model_name.replace('/', '_')}{hf_train_version}"

# --- Process each model ---
for model_name in models_to_convert:
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Setup directories
    safe_model_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(output_base_dir, safe_model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    tflite_model_path = os.path.join(model_output_dir, f"{safe_model_name}.tflite")
    vocab_path = os.path.join(model_output_dir, "vocab.txt")

    # Fast-skip already completed model outputs.
    if os.path.exists(tflite_model_path) and os.path.exists(vocab_path):
        print("✓ Skipping (already exists): tflite + vocab.txt")
        continue

    stale_temp_saved_model = os.path.join(model_output_dir, "temp_saved_model")
    if os.path.exists(stale_temp_saved_model):
        shutil.rmtree(stale_temp_saved_model)
    
    # Determine model source (local priority)
    local_model_path = get_local_model_path(model_name)
    if local_model_path:
        model_source = local_model_path
        print(f"✓ Using local model: {model_source}")
    else:
        model_source = get_hf_model_path(model_name)
        print(f"✓ Using HuggingFace model: {model_source}")
    
    # --- Step 1: Load model and tokenizer ---
    try:
        print("Step 1: Loading model and tokenizer...")
        model = load_tf_model(model_source, hf_token)
        tokenizer_kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        continue
    
    # --- Step 2: Export vocab.txt ---
    try:
        print("Step 2: Exporting vocab.txt...")

        # Get vocab from tokenizer and save to vocab.txt
        vocab = tokenizer.get_vocab()
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        
        print(f"  ✓ Vocab.txt saved to {vocab_path}")
    except Exception as e:
        print(f"  ✗ Error saving vocab: {e}")
        continue
    
    # --- Step 3: Export as SavedModel ---
    try:
        print("Step 3: Exporting as SavedModel...")
        temp_saved_model_path = os.path.join(model_output_dir, "temp_saved_model")
        model.save_pretrained(temp_saved_model_path, saved_model=True)
        print(f"  ✓ SavedModel exported")
    except Exception as e:
        print(f"  ✗ Error exporting SavedModel: {e}")
        if os.path.exists(temp_saved_model_path):
            shutil.rmtree(temp_saved_model_path)
        continue
    
    # --- Step 4: Convert to TFLite with int8 quantization ---
    try:
        print("Step 4: Converting to TFLite...")
        tf_saved_model_path = os.path.join(temp_saved_model_path, "saved_model", "1")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
        
        if not args.no_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        
        tflite_model = converter.convert()
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"  ✓ TFLite model saved to {tflite_model_path}")
    except Exception as e:
        print(f"  ✗ Error converting to TFLite: {e}")
    finally:
        # Cleanup temporary SavedModel
        if os.path.exists(temp_saved_model_path):
            shutil.rmtree(temp_saved_model_path)
    
    print(f"✓ Completed: {model_name}")

print(f"\n{'='*60}")
print(f"All models processed! Output saved to: {output_base_dir}/")
print(f"{'='*60}")