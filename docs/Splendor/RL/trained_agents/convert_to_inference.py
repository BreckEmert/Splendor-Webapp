# RL/trained_agents/convert_to_inference.py
# How to run (Bash or PowerShell):
#   cd "F:/GitHub/Splendor-AI/Splendor/RL/trained_agents"
#   # Bash (Git Bash/WSL):
#   python convert_to_inference.py inference_model.keras
#   # PowerShell / cmd:
#   py convert_to_inference.py inference_model.keras
#
# If no argument is given, defaults to "inference_model.keras" in this folder.

"""Strips optimizer/training data from a Keras model and exports Dense weights to .npz."""

from pathlib import Path
import numpy as np

# This script is an offline utility: it uses TF only to read the .keras file.
import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore


def main():
    script_dir = Path(__file__).resolve().parent
    # Input: your trained Keras model (Functional or Sequential)
    model_path = script_dir / "07-20-00-51__512-512-256.keras"
    # Output: NumPy weights for the NumPy-only inference model
    export_path = model_path.with_suffix(".npz")

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)  # type: ignore

    # Collect Dense layers in model order; export as W1,b1,W2,b2,...
    W_keys, B_keys, arrays = [], [], []
    dense_idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            dense_idx += 1
            W, b = layer.get_weights()  # [kernel, bias]
            W = W.astype(np.float32, copy=False)
            b = b.astype(np.float32, copy=False)

            W_key = f"W{dense_idx}"
            b_key = f"b{dense_idx}"
            W_keys.append(W_key)
            B_keys.append(b_key)
            arrays.append((W_key, W))
            arrays.append((b_key, b))
            print(f"  {W_key}: {W.shape}   {b_key}: {b.shape}")

    if dense_idx == 0:
        raise RuntimeError("No Dense layers found in model—nothing to export.")

    # Save in a simple, inference-friendly format
    np.savez_compressed(export_path, **{k: v for k, v in arrays})
    print(f"Saved inference weights to {export_path}")

    # Optional: also save a tiny metadata text file for human inspection
    meta_path = export_path.with_suffix(".txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("Exported Dense layers (W: [in,out], b: [out])\n")
        for (Wk, Bk) in zip(W_keys, B_keys):
            W = next(a for k, a in arrays if k == Wk)
            b = next(a for k, a in arrays if k == Bk)
            f.write(f"{Wk}: {tuple(W.shape)}, {Bk}: {tuple(b.shape)}\n")
    print(f"Wrote shapes to {meta_path}")


if __name__ == "__main__":
    main()
