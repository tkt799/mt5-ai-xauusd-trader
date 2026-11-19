"""Tests for checkpoint-driven input-dimension inference in `LSTMModel`.

These do not need PyTorch: `infer_input_dim_from_state_dict` only inspects shapes,
which numpy arrays provide just as well as tensors. The regression they lock down is
that `main.py` builds `LSTMModel` without passing `input_dim`, while
`scripts/train_models.py` saves a checkpoint shaped by the real feature count
(248 columns with multi-timeframe features). The old default of 140 made
`load_state_dict` fail, which silently downgraded the bot to HOLD-only trading.
"""

from __future__ import annotations

import numpy as np

from src.models.lstm_model import DEFAULT_INPUT_DIM, infer_input_dim_from_state_dict


def test_infers_dimension_from_standard_lstm_checkpoint() -> None:
    state = {
        "lstm.weight_ih_l0": np.zeros((4 * 64, 248)),
        "lstm.weight_hh_l0": np.zeros((4 * 64, 64)),
        "fc.weight": np.zeros((3, 64)),
    }
    assert infer_input_dim_from_state_dict(state) == 248


def test_infers_dimension_from_attention_checkpoint() -> None:
    state = {"input_projection.weight": np.zeros((128, 13))}
    assert infer_input_dim_from_state_dict(state) == 13


def test_prefers_lstm_weight_when_both_present() -> None:
    state = {
        "input_projection.weight": np.zeros((128, 13)),
        "lstm.weight_ih_l0": np.zeros((256, 247)),
    }
    assert infer_input_dim_from_state_dict(state) == 247


def test_returns_none_for_unusable_input() -> None:
    assert infer_input_dim_from_state_dict(None) is None
    assert infer_input_dim_from_state_dict({}) is None
    assert infer_input_dim_from_state_dict({"unrelated": np.zeros((3,))}) is None


def test_default_dimension_is_the_legacy_value() -> None:
    # The constant is still used as the fallback for brand-new, untrained models.
    assert DEFAULT_INPUT_DIM == 140
