import tensorflow as tf
import os
import yaml

try:
    from transformers import TFAutoModelForSequenceClassification
except ImportError as import_error:
    raise RuntimeError(
        "TensorFlow model classes are not available in your installed 'transformers' package. "
        "Use a TensorFlow-compatible transformers version, e.g.:\n"
        "  pip install 'transformers<5'\n"
        "Then rerun this script."
    ) from import_error

def _parse_version(version_text: str):
    core = version_text.split("+")[0]
    parts = core.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return major, minor, patch


if _parse_version(tf.__version__) < (2, 14, 0):
    raise RuntimeError(
        f"This script requires TensorFlow >=2.14.0, but found {tf.__version__}."
    )

if tf.__version__ != "2.14.0":
    print(
        f"Warning: script was originally tested on TensorFlow 2.14.0; found {tf.__version__}."
    )


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def find_best_checkpoint(base_path: str):
    """Find the numerical checkpoint with highest step number in a directory."""
    if not os.path.exists(base_path):
        return None
    try:
        checkpoints = [
            d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith("checkpoint-")
        ]
        if not checkpoints:
            return None
        # Sort by checkpoint step number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        return os.path.join(base_path, checkpoints[0])
    except Exception:
        return None


def export_model_to_tflite(model_path: str, safe_model_name: str, fallback_hf_path: str = None, checkpoint_path: str = None):
    """Convert a PyTorch/Keras model to TFLite format.
    
    Args:
        model_path: Local path to model directory or HuggingFace Hub model ID
        safe_model_name: Sanitized model name for output file naming
        fallback_hf_path: HuggingFace Hub model ID to try if local path fails
        checkpoint_path: Checkpoint directory to try as last resort
    """
    saved_model_path = f"saved_model_{safe_model_name}"
    tflite_model_path = f"{safe_model_name}.tflite"

    loaded_from = None
    try:
        print(f"  Loading model from: {model_path}")
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            from_pt=True,
        )
        loaded_from = model_path
    except Exception as e:
        local_err = str(e)
        hf_err = None
        
        if fallback_hf_path:
            print(f"  Local incomplete. Trying HuggingFace Hub...")
            try:
                model = TFAutoModelForSequenceClassification.from_pretrained(
                    fallback_hf_path,
                    from_pt=True,
                )
                loaded_from = "HuggingFace Hub"
            except Exception as hf_e:
                hf_err = str(hf_e)
                
                # Try checkpoint as last resort
                if checkpoint_path and os.path.exists(checkpoint_path):
                    print(f"  HuggingFace unavailable. Trying checkpoint: {checkpoint_path}")
                    try:
                        model = TFAutoModelForSequenceClassification.from_pretrained(
                            checkpoint_path,
                            from_pt=True,
                        )
                        loaded_from = checkpoint_path
                    except Exception as ckpt_e:
                        raise RuntimeError(
                            f"Failed to load model from all sources:\n"
                            f"1. Local: {local_err}\n"
                            f"2. HuggingFace: {hf_err}\n"
                            f"3. Checkpoint: {str(ckpt_e)}"
                        ) from ckpt_e
                else:
                    raise RuntimeError(
                        f"Failed to load from local and HuggingFace.\n"
                        f"Local: {local_err}\n"
                        f"HuggingFace: {hf_err}\n"
                        f"No checkpoint directory found to try as fallback."
                    ) from hf_e
        else:
            raise
    
    print(f"  ✓ Loaded from {loaded_from}")
    
    print(f"  Exporting SavedModel...")
    model.save_pretrained(saved_model_path, saved_model=True)
    print(f"  SavedModel exported to {saved_model_path}")

    print(f"  Converting to TFLite...")
    tf_saved_model_path = os.path.join(saved_model_path, "saved_model", "1")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as model_file:
        model_file.write(tflite_model)

    print(f"  ✓ TFLite model saved to {tflite_model_path}")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = load_config(config_path)

    hf_username = config["hf_username"]
    hf_train_version = config["hf_train_version"]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for model_name in config["models"]:
        safe_model_name = model_name.replace("/", "_")
        
        # Try local checkpoint in src/output/ first
        local_model_path = os.path.join(
            script_dir, "output", f"{safe_model_name}{hf_train_version}"
        )
        
        # Find best checkpoint as fallback
        best_checkpoint = find_best_checkpoint(local_model_path)
        
        # HuggingFace fallback path
        fallback_hf_path = f"{hf_username}{safe_model_name}{hf_train_version}"
        
        print(f"\n{model_name}:")
        print(f"  Converting...")
        try:
            export_model_to_tflite(
                local_model_path, 
                safe_model_name, 
                fallback_hf_path=fallback_hf_path,
                checkpoint_path=best_checkpoint
            )
        except Exception as e:
            error_msg = str(e)[:500]
            print(f"  ✗ Error: {error_msg}")
            
            # If all sources failed, provide helpful guidance
            if "Failed to load model from all sources" in str(e):
                print("\n  ⚠️  SOLUTION: Your trained models don't have weights saved locally or on HuggingFace Hub.")
                print("     To convert models to TFLite, you need to:")
                print("     1. Save the best trained model locally by modifying train.py or manually:")
                print(f"        trainer.save_model('{local_model_path}')")
                print("     2. OR ensure models are properly pushed to your HuggingFace Hub account")
                print(f"        (currently trying to load from: {fallback_hf_path})")