# Splendor/RL/inference_model.py

import numpy as np


class InferenceModel:
    """Numpy-only MLP for inference"""

    def __init__(self, weights_path: str, leaky_slope: float = 0.3):
        """Load weights from an .npz"""
        # keys look like W1,b1,W2,b2,... (W are [in,out], b are [out])
        data = np.load(weights_path, allow_pickle=False)
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
        x = state.astype(np.float32)
        for i in range(len(self.W) - 1):
            x = x @ self.W[i] + self.b[i]
            x = np.where(x > 0, x, self.leaky * x)
        x = x @ self.W[-1] + self.b[-1]
        return x

    def get_predictions(self, state: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        qs = self._forward(state)
        qs = qs.astype(np.float32, copy=False)
        qs[~legal_mask] = -np.inf
        return qs
