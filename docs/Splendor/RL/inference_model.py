# Splendor/RL/inference_model.py

import numpy as np


class InferenceModel:
    """Numpy-only MLP for inference."""

    def __init__(self, weights_path: str, leaky_slope: float = 0.3):
        """Load weights from an .npz file"""
        data = np.load(weights_path, allow_pickle=False)

        # Single-head weights
        if "W1" in data and "b1" in data:
            Ws, bs, i = [], [], 1
            while f"W{i}" in data and f"b{i}" in data:
                Ws.append(data[f"W{i}"].astype(np.float32))
                bs.append(data[f"b{i}"].astype(np.float32))
                i += 1
            if not Ws:
                raise ValueError("No weights found in weights file")

            self.shared_W, self.shared_b = Ws[:-1], bs[:-1]
            self.W_out, self.b_out = Ws[-1], bs[-1]
            self.is_dueling = False

        # Dueling or other
        else:
            dense_nums = sorted(
                [int(k[len("dense") : k.index("_W")]) for k in data.files
                 if k.startswith("dense") and k.endswith("_W")]
            )
            self.shared_W = [data[f"dense{n}_W"].astype(np.float32) for n in dense_nums]
            self.shared_b = [data[f"dense{n}_b"].astype(np.float32) for n in dense_nums]

            has_adv = "advantage_W" in data and "advantage_b" in data
            has_val = "value_W" in data and "value_b" in data
            self.is_dueling = has_adv and has_val

            if self.is_dueling:
                self.W_adv = data["advantage_W"].astype(np.float32)
                self.b_adv = data["advantage_b"].astype(np.float32)
                self.W_val = data["value_W"].astype(np.float32)
                self.b_val = data["value_b"].astype(np.float32)

            else:
                if "q_W" in data and "q_b" in data:
                    self.W_out = data["q_W"].astype(np.float32)
                    self.b_out = data["q_b"].astype(np.float32)
                else:
                    self.W_out = self.shared_W.pop().astype(np.float32)
                    self.b_out = self.shared_b.pop().astype(np.float32)

        self.leaky = leaky_slope

        # Model metadata
        self.state_dim = self.shared_W[0].shape[0]
        self.layer_sizes = [w.shape[1] for w in self.shared_W]
        self.action_dim = (self.W_adv.shape[1] if self.is_dueling else self.W_out.shape[1])

    def _forward(self, state: np.ndarray) -> np.ndarray:
        x = state.astype(np.float32)
        for W, b in zip(self.shared_W, self.shared_b):
            x = x @ W + b
            x = np.where(x > 0, x, self.leaky * x)

        if self.is_dueling:
            A = x @ self.W_adv + self.b_adv
            V = x @ self.W_val + self.b_val
            A = A - A.mean(axis=-1, keepdims=True)
            return V + A

        return x @ self.W_out + self.b_out

    def get_predictions(self, state: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        # Q-values
        qs = self._forward(state)
        qs = qs.astype(np.float32, copy=False)
        qs[~legal_mask] = -np.inf
        return qs
