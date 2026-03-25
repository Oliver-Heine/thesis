import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import os

if tf.__version__ != "2.14.0":
    raise RuntimeError(
        f"This script requires TensorFlow 2.14.0, but found {tf.__version__}."
    )

# --- Paths ---
hf_model_name_or_path = "OliverHeine/distilbert-base-uncased_train_v2"  # e.g., "./distilbert-finetuned"
saved_model_path = "saved_model"
tflite_model_path = "distilbert_model.tflite"

# --- Step 1: Load model and tokenizer ---
model = TFDistilBertForSequenceClassification.from_pretrained(hf_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)

# --- Step 2: Export as SavedModel ---
model.save_pretrained(saved_model_path, saved_model=True)
print(f"SavedModel exported to {saved_model_path}")

# --- Step 3: Convert to TFLite with int8 quantization ---
tf_saved_model_path = os.path.join(saved_model_path, "saved_model", "1")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

tflite_model = converter.convert()
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")