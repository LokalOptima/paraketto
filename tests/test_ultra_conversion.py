"""Numerical checks for the transformations required by the Ultra converter.

Run: uv run python -m unittest discover -s tests -p test_ultra_conversion.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_ultra_weights import BATCH_NORM_EPS, fold_batch_norm, reorder_lstm_gates


class UltraConversionTests(unittest.TestCase):
    def test_folded_convolution_matches_batch_norm_at_edges(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(3, 17)).astype(np.float32)
        w = rng.normal(size=(3, 1, 9)).astype(np.float32)
        gamma, beta, mean = rng.normal(size=(3, 3)).astype(np.float32)
        var = rng.uniform(0.01, 3, size=3).astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, ((0, 0), (4, 4))), 9, axis=1)
        conv = np.einsum("ctk,ck->ct", windows, w[:, 0, :])
        reference = (conv - mean[:, None]) / np.sqrt(var[:, None] + BATCH_NORM_EPS)
        reference = reference * gamma[:, None] + beta[:, None]
        folded_w, folded_b = fold_batch_norm(w, gamma, beta, mean, var)
        actual = np.einsum("ctk,ck->ct", windows, folded_w[:, 0, :]) + folded_b[:, None]
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)

    def test_converted_lstm_matches_recurrent_step(self):
        rng = np.random.default_rng(31)
        h, inputs = 5, 7
        w_ih = rng.normal(size=(4 * h, inputs))
        w_hh = rng.normal(size=(4 * h, h))
        b_ih, b_hh = rng.normal(size=(2, 4 * h))
        state, cell = rng.normal(size=(2, h))
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        other_state, other_cell = state.copy(), cell.copy()
        for x in rng.normal(size=(6, inputs)):
            i, f, g, o = np.split(w_ih @ x + b_ih + w_hh @ state + b_hh, 4)
            cell = sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            state = sigmoid(o) * np.tanh(cell)
            gates = (reorder_lstm_gates(w_ih) @ x + reorder_lstm_gates(w_hh) @ other_state
                     + reorder_lstm_gates(b_ih) + reorder_lstm_gates(b_hh))
            i, o, f, g = np.split(gates, 4)
            other_cell = sigmoid(f) * other_cell + sigmoid(i) * np.tanh(g)
            other_state = sigmoid(o) * np.tanh(other_cell)
            np.testing.assert_allclose(other_state, state, atol=1e-14)
            np.testing.assert_allclose(other_cell, cell, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
