import argparse
import re
import numpy as np
import tensorflow as tf
from urllib.parse import urlparse
from transformers import AutoTokenizer

TFLITE_PATH = "distilbert_model.tflite"
TOKENIZER_PATH = "OliverHeine/distilbert-base-uncased_train_v2"


def normalize_url(url: str) -> str:
    url = url.lower().strip()

    url_clean = re.sub(r"^https?://", "", url)
    url_clean = re.sub(r"^www\.", "", url_clean)

    parsed = urlparse("http://" + url_clean)
    host = parsed.netloc.split(":")[0]
    host_parts = [p for p in host.split(".") if p]

    tokens = []

    subdomain_parts = host_parts[:-2] if len(host_parts) > 2 else []
    domain_part = host_parts[-2] if len(host_parts) >= 2 else (host_parts[0] if host_parts else "")
    suffix_parts = host_parts[-1:] if len(host_parts) >= 2 else []

    if subdomain_parts:
        tokens.append("<subdomain>")
        tokens.extend(subdomain_parts)

    tokens.append("<domain>")
    tokens.append(domain_part)

    tokens.append("<suffix>")
    tokens.extend(suffix_parts)

    if parsed.path and parsed.path != "/":
        tokens.append("<path>")
        tokens.extend(re.split(r"[/\-_.?=&]", parsed.path))

    if parsed.query:
        tokens.append("<query>")
        tokens.extend(re.split(r"[=&]", parsed.query))

    return " ".join([t for t in tokens if t])


def build_runner():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()

    def run_inference(normalized_text: str):
        encoded = tokenizer(
        normalized_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

        input_details = interpreter.get_input_details()

        # Match dynamic input shapes.
        seq_len = encoded["input_ids"].shape[1]
        for inp in input_details:
            interpreter.resize_tensor_input(inp["index"], [1, seq_len], strict=False)
        interpreter.allocate_tensors()

        name_to_array = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # Set tensors by matching tensor name suffix.
        for inp in interpreter.get_input_details():
            for key, arr in name_to_array.items():
                if key in inp["name"]:
                    interpreter.set_tensor(inp["index"], arr)
                    break

        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]["index"])
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        pred = int(np.argmax(probs))
        return pred, probs

    return run_inference


def format_result(pred: int, probs: np.ndarray) -> str:
    label = "malicious" if pred == 1 else "benign"
    confidence = float(probs[pred])
    return f"{confidence:.6f} {label}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run URL inference with a TFLite model")
    parser.add_argument("--url", type=str, help="Single URL to classify")
    args = parser.parse_args()

    run_inference = build_runner()

    if args.url:
        normalized = normalize_url(args.url)
        pred, probs = run_inference(normalized)
        print(format_result(pred, probs))
    else:
        print("Enter URL (type 'exit' to quit)")
        while True:
            url = input("URL> ").strip()
            if not url:
                continue
            if url.lower() == "exit":
                break

            normalized = normalize_url(url)
            pred, probs = run_inference(normalized)
            print(format_result(pred, probs))
