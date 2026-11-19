"""Unit tests for the data-fetch and model-training helper scripts.

These cover the pure data-shaping pieces of `scripts/fetch_history.py` and
`scripts/train_models.py`; the MT5 download and the torch training loop are
exercised manually (see docs/TRAINING_PIPELINE.md).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.fetch_history import _normalise
from scripts.train_models import _labels_from_forward, load_ohlcv


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": ["2024-01-02 00:05", "2024-01-01 00:00", "2024-01-01 00:00"],
            "Open": [2.0, 1.0, 1.5],
            "High": [2.1, 1.2, 1.6],
            "Low": [1.9, 0.9, 1.4],
            "Close": [2.0, 1.1, 1.5],
            "Tick_Volume": [10, 20, 30],
        }
    )


def test_normalise_lowercases_sorts_and_dedupes() -> None:
    result = _normalise(_frame())

    assert list(result.columns)[:5] == ["time", "open", "high", "low", "close"]
    assert result["time"].is_monotonic_increasing
    assert len(result) == 2
    # The last duplicate timestamp wins.
    assert result.iloc[0]["close"] == pytest.approx(1.5)


def test_normalise_drops_non_positive_close_and_keeps_volume() -> None:
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00", "2024-01-01 00:05"],
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [0.0, 1.0],
        }
    )

    result = _normalise(frame)

    assert len(result) == 1
    assert "tick_volume" in result.columns
    assert result.iloc[0]["close"] == pytest.approx(1.0)


def test_labels_map_extremes_to_buy_and_sell() -> None:
    forward = pd.Series([-0.01, -0.001, 0.0, 0.001, 0.01], dtype="float64")

    labels = _labels_from_forward(forward, 0.2)

    # 0 = HOLD, 1 = BUY, 2 = SELL (ModelAction ordering).
    assert labels.tolist() == [2, 0, 0, 0, 1]


def test_labels_respect_quantile_width() -> None:
    forward = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype="float64")

    labels = _labels_from_forward(forward, 0.33)

    assert set(labels.tolist()) == {0, 1, 2}
    assert (labels == 1).sum() >= 1
    assert (labels == 2).sum() >= 1


def test_load_ohlcv_cleans_csv(tmp_path) -> None:
    path = tmp_path / "bars.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"],
            "open": [1.0, 1.1, 1.2],
            "high": [1.05, 1.15, 1.25],
            "low": [0.95, 1.05, 1.15],
            "close": [1.0, 0.0, 1.3],
            "tick_volume": [10, 20, 30],
        }
    ).to_csv(path, index=False)

    frame = load_ohlcv(path)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == 2
    assert (frame["close"] > 0).all()
    assert "tick_volume" in frame.columns


def test_load_ohlcv_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "broken.csv"
    pd.DataFrame({"time": ["2024-01-01"], "open": [1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_ohlcv(path)
