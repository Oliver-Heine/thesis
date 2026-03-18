#!/usr/bin/env python3
"""
Convert a Hugging Face DistilBERT ONNX model to TensorFlow Lite for Android
"""

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import argparse
import os

def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, use_quantization: bool = True):
    # -------------------------
    # 1. Load ONNX model
    # -------------------------
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    print(f"Loading ONNX model from {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # -------------------------
    # 2. Convert ONNX -> TensorFlow SavedModel
    # -------------------------
    print("Converting ONNX model to TensorFlow SavedModel...")
    tf_rep = prepare(onnx_model)
    saved_model_dir = "tmp_saved_model"
    tf_rep.export_graph(saved_model_dir)
    print(f"SavedModel exported to {saved_model_dir}/")

    # -------------------------
    # 3. Convert SavedModel -> TFLite
    # -------------------------
    print("Converting SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if use_quantization:
        print("Applying FP16 quantization for mobile...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved at {tflite_path}")

    # -------------------------
    # 4. Optional: Inspect TFLite input/output
    # -------------------------
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("TFLite input details:", interpreter.get_input_details())
    print("TFLite output details:", interpreter.get_output_details())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX DistilBERT model to TFLite")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--tflite_path", type=str, default="distilbert.tflite", help="Path to save TFLite model")
    parser.add_argument("--no_quant", action="store_true", help="Disable FP16 quantization")
    args = parser.parse_args()

    convert_onnx_to_tflite(args.onnx_path, args.tflite_path, use_quantization=not args.no_quant)
