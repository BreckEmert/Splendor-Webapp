# Splendor/RL/inference_model.py

import numpy as np


class InferenceModel:
    """
    Minimal numpy-only MLP for inference.
    Expects a 3-layer MLP (2 hidden + 1 linear action head) but works with any depth.
    """

    def __init__(self, weights_path: str, leaky_slope: float = 0.3):
        # Load weights from an .npz:
        # keys: W1,b1,W2,b2,...,Wk,bk (W are [in,out], b are [out])
        data = np.load(weights_path, allow_pickle=False)
        # Collect in order W1..Wk, b1..bk
        Ws, bs = [], []
        i = 1
        while f"W{i}" in data and f"b{i}" in data:
            Ws.append(data[f"W{i}"].astype(np.float32))
            bs.append(data[f"b{i}"].astype(np.float32))
            i += 1
        if not Ws:
            raise ValueError("No weights found in weights file")

        self.W = Ws
        self.b = bs
        self.leaky = leaky_slope

        # Shapes / metadata
        self.state_dim = self.W[0].shape[0]
        self.layer_sizes = [w.shape[1] for w in self.W[:-1]]
        self.action_dim = self.W[-1].shape[1]

    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Single-sample forward pass."""
        x = state.astype(np.float32)

        # Dense hidden layers with LeakyReLU(0.3)
        for i in range(len(self.W) - 1):
            x = x @ self.W[i] + self.b[i]
            x = np.where(x > 0, x, self.leaky * x)

        # Linear action head
        x = x @ self.W[-1] + self.b[-1]
        return x

    def get_predictions(self, state: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Returns q-values"""
        qs = self._forward(state)
        qs = qs.astype(np.float32, copy=False)
        qs[~legal_mask] = -np.inf
        return qs
