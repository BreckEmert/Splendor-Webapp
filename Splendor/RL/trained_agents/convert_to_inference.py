# RL/trained_agents/convert_to_inference.py
# How to run (Bash or PowerShell):
#   cd "F:/GitHub/Splendor-AI/Splendor/RL/trained_agents"
#   python convert_to_inference.py inference_model.keras
#
# If no argument is given, defaults to "inference_model.keras" in this folder.

"""Strips optimizer/training data from a Keras model and exports Dense weights to .npz."""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf  # OFFLINE so this is ok
from tensorflow.keras.layers import Dense  # type: ignore


def iter_dense_layers(m):
    """Recursively get layers from sequential/functional/wrappers."""
    for layer in getattr(m, "layers", []):
        if isinstance(layer, Dense):
            yield layer
        if hasattr(layer, "layers"):
            yield from iter_dense_layers(layer)

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: convert_to_inference.py <model.keras>")
    model_path = Path(sys.argv[1]).resolve()
    export_path = model_path.with_suffix(".npz")

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)  # type: ignore

    # Collect Dense layers and export as W1,b1,W2,b2,...
    W_keys, B_keys, arrays = [], [], []
    dense_idx = 0
    for layer in iter_dense_layers(model):  # nested models
        if isinstance(layer, Dense):
            dense_idx += 1
            weights = layer.get_weights()
            W = np.asarray(weights[0], dtype=np.float32)
            b = (np.asarray(weights[1], dtype=np.float32)
                 if len(weights) > 1 else np.zeros(layer.units, np.float32))

            W_key = f"{layer.name}_W"
            b_key = f"{layer.name}_b"

            W_keys.append(W_key)
            B_keys.append(b_key)
            arrays.append((W_key, W))
            arrays.append((b_key, b))
            print(f"{W_key}: {W.shape}    {b_key}: {b.shape}")

    if dense_idx == 0:
        raise RuntimeError("No Dense layers; nothing to export.")

    # Save as npz
    np.savez_compressed(export_path, **{k: v for k, v in arrays})
    print(f"Saved inference weights to {export_path}")

    # Save the metadata
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
