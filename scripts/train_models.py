"""
MT5 AI/ML Trading Bot - Enterprise Edition
scripts/train_models.py
Train the PPO and/or LSTM model from a historical CSV exported by fetch_history.py.

Artifacts are written with the names main.py expects:
    models/trained/ppo_{symbol}.zip    (PPO, Stable-Baselines3)
    models/trained/lstm_{symbol}.pt    (LSTM state_dict, PyTorch)

Examples
--------
  # Everything with defaults (PPO 200k steps + LSTM 10 epochs)
  python scripts/train_models.py --data data/raw/XAUUSD_M5_20240101_20251231.csv

  # Only the LSTM, quicker settings
  python scripts/train_models.py --data latest --model lstm --lstm-epochs 3 --max-samples 5000

  # Validate the dataset/feature pipeline without training
  python scripts/train_models.py --data latest --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("train_models")

MIN_PPO_BARS = 600
MIN_LSTM_BARS = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO/LSTM models on historical OHLCV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="latest",
        help="CSV from scripts/fetch_history.py, or 'latest' for the newest file in data/raw.",
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol used in artifact names.")
    parser.add_argument(
        "--timeframe", default="M5", help="Base timeframe of the CSV (used for features)."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["ppo", "lstm", "all"],
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models" / "trained",
        help="Directory for trained artifacts.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Training report JSON. Default: reports/training_report_<timestamp>.json",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data only, do not train.")

    lstm = parser.add_argument_group("LSTM")
    lstm.add_argument("--window", type=int, default=60, help="Sequence length in bars.")
    lstm.add_argument("--horizon", type=int, default=12, help="Label look-ahead in bars.")
    lstm.add_argument(
        "--label-quantile",
        type=float,
        default=0.33,
        help="Bottom/top quantile of forward returns mapped to SELL/BUY.",
    )
    lstm.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden size.")
    lstm.add_argument("--num-layers", type=int, default=2, help="Stacked LSTM layers.")
    lstm.add_argument("--lstm-epochs", type=int, default=10, help="Training epochs.")
    lstm.add_argument("--lstm-lr", type=float, default=1e-3, help="Learning rate.")
    lstm.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    lstm.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Random subsample of sequences (0 = use all available).",
    )
    lstm.add_argument(
        "--train-split", type=float, default=0.8, help="Chronological train/validation split."
    )
    lstm.add_argument(
        "--no-mtf",
        action="store_true",
        help="Disable multi-timeframe features (faster, fewer columns).",
    )

    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--ppo-window", type=int, default=20, help="Observation window in bars.")
    ppo.add_argument("--ppo-steps", type=int, default=200_000, help="Total training timesteps.")
    ppo.add_argument("--ppo-verbose", type=int, default=0, help="SB3 verbosity (0 = quiet).")
    ppo.add_argument(
        "--initial-balance", type=float, default=10000.0, help="Simulated starting balance."
    )
    ppo.add_argument("--spread", type=float, default=0.20, help="Simulated spread (USD).")
    ppo.add_argument("--slippage", type=float, default=0.05, help="Simulated slippage (USD).")

    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_csv(data: str) -> Path:
    if data.lower() != "latest":
        path = Path(data)
        return path if path.is_absolute() else (ROOT / path)

    raw_dir = ROOT / "data" / "raw"
    candidates = sorted(raw_dir.glob("*.csv"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Run scripts/fetch_history.py first.")
    return candidates[-1]


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load a CSV and return a clean, time-indexed OHLCV frame."""
    frame = pd.read_csv(path)
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame = frame.dropna(subset=["time"]).set_index("time")
    elif not isinstance(frame.index, pd.RangeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[frame.index.notna()]

    missing = [column for column in ("open", "high", "low", "close") if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = frame["volume"] if "volume" in frame.columns else 1.0

    numeric = ["open", "high", "low", "close", "tick_volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=numeric)
    frame = frame[frame["close"] > 0]
    return frame.sort_index()


def _sequence_dataset(
    torch_module,
    features: np.ndarray,
    labels: np.ndarray,
    window: int,
):
    """Lazily slice windows so large CSVs never materialise in RAM."""

    class _SequenceDataset(torch_module.utils.data.Dataset):
        def __len__(self) -> int:
            return max(0, len(labels) - window + 1)

        def __getitem__(self, index: int):
            chunk = features[index : index + window]
            target = labels[index + window - 1]
            return (
                torch_module.from_numpy(chunk).float(),
                torch_module.tensor(target, dtype=torch_module.long),
            )

    return _SequenceDataset()


def _labels_from_forward(forward: pd.Series, quantile: float) -> np.ndarray:
    """Map forward log returns to 3 classes: 0=HOLD, 1=BUY, 2=SELL."""
    values = forward.to_numpy(dtype="float64")
    low = float(np.nanquantile(values, quantile))
    high = float(np.nanquantile(values, 1.0 - quantile))
    labels = np.zeros(len(values), dtype="int64")
    labels[values > high] = 1
    labels[values < low] = 2
    return labels


def train_lstm(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    from src.core.feature_engineering import FeatureEngineer

    if len(frame) < MIN_LSTM_BARS:
        raise RuntimeError(
            f"Only {len(frame)} bars available, need at least {MIN_LSTM_BARS} for LSTM training."
        )

    base_tf = args.timeframe.upper()
    logger.info("Engineering features (base timeframe %s) ...", base_tf)
    engineer = FeatureEngineer(
        base_timeframe=base_tf,
        timeframes=[base_tf] if args.no_mtf else None,
        normalize=True,
    )
    features = engineer.compute_features(frame.copy(), drop_ohlcv=True)
    if features.empty:
        raise RuntimeError("Feature engineering returned no rows; check the input CSV columns.")

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < args.window + args.horizon + 50:
        raise RuntimeError(f"Only {len(features)} complete feature rows; need more history.")

    # Align raw prices with the (possibly shorter) feature matrix, then drop the
    # tail that has no forward label yet.
    frame = frame.reindex(features.index).ffill()
    cutoff = len(frame) - args.horizon
    if cutoff <= args.window:
        raise RuntimeError("Not enough bars after the label horizon; fetch more history.")

    features = features.iloc[:cutoff]
    close = frame["close"].astype("float64").iloc[:cutoff]
    forward = np.log(close.shift(-args.horizon) / close).iloc[:cutoff]
    labels = _labels_from_forward(forward, args.label_quantile)

    feature_values = features.to_numpy(dtype="float32")
    total = max(0, len(labels) - args.window + 1)
    if total < 2:
        raise RuntimeError("Not enough sequences for training; fetch more history.")

    split = max(1, min(total - 1, int(total * args.train_split)))
    distribution = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=False)
    }
    class_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    readable = {class_names[key]: value for key, value in distribution.items()}

    logger.info(
        "LSTM dataset: features=%d, sequences=%d (train=%d, val=%d), labels=%s",
        feature_values.shape[1],
        total,
        split,
        total - split,
        readable,
    )

    if args.dry_run:
        return {
            "dry_run": True,
            "features": int(feature_values.shape[1]),
            "sequences": int(total),
            "train_sequences": int(split),
            "validation_sequences": int(total - split),
            "label_distribution": readable,
        }

    if not _has_module("torch"):
        raise RuntimeError(
            "PyTorch is not installed. Run: "
            "venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.models.lstm_model import LSTMModel

    dataset = _sequence_dataset(torch, feature_values, labels, args.window)

    if args.max_samples and total > args.max_samples:
        generator = np.random.default_rng(args.seed)
        chosen = np.sort(generator.choice(total, size=args.max_samples, replace=False))
        train_indices = [int(index) for index in chosen if index < split]
        val_indices = [int(index) for index in chosen if index >= split]
        logger.info("Subsampled %d of %d sequences.", len(chosen), total)
    else:
        train_indices = list(range(split))
        val_indices = list(range(split, total))

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Train or validation split is empty; adjust --train-split/--max-samples."
        )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    torch.manual_seed(args.seed)
    model = LSTMModel(
        input_dim=feature_values.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )
    started = time.perf_counter()
    model.train(train_loader, epochs=args.lstm_epochs, learning_rate=args.lstm_lr)
    elapsed = time.perf_counter() - started

    out_path = args.out_dir / f"lstm_{args.symbol.lower()}.pt"
    model.save(out_path)

    accuracy = _evaluate_lstm(model, val_loader, torch)
    logger.info("LSTM saved -> %s (%.1fs, val accuracy=%s)", out_path, elapsed, accuracy)
    return {
        "artifact": str(out_path),
        "features": int(feature_values.shape[1]),
        "sequences": int(total),
        "train_sequences": len(train_indices),
        "validation_sequences": len(val_indices),
        "label_distribution": readable,
        "epochs": args.lstm_epochs,
        "seconds": round(elapsed, 1),
        "val_accuracy": accuracy,
    }


def _evaluate_lstm(model, loader, torch_module) -> float | None:
    net = getattr(model, "model", None)
    if net is None:
        return None
    device = model.device
    net.eval()
    correct = 0
    seen = 0
    with torch_module.no_grad():
        for batch_features, batch_labels in loader:
            logits = net(batch_features.to(device))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch_labels.to(device)).sum().item())
            seen += int(batch_labels.size(0))
    return round(correct / seen, 4) if seen else None


def train_ppo(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    if not _has_module("torch") or not _has_module("stable_baselines3"):
        raise RuntimeError(
            "PPO training needs torch + stable-baselines3. "
            "Run: venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )

    from src.models.ppo_agent import PPOAgent
    from src.trading.trading_env import TradingEnv

    if len(frame) < MIN_PPO_BARS:
        raise RuntimeError(
            f"Only {len(frame)} bars available, need at least {MIN_PPO_BARS} for PPO training."
        )

    env_frame = (
        frame[["open", "high", "low", "close", "tick_volume"]]
        .rename(columns={"tick_volume": "volume"})
        .astype("float32")
    )

    logger.info(
        "Training PPO: bars=%d, window=%d, timesteps=%d",
        len(env_frame),
        args.ppo_window,
        args.ppo_steps,
    )
    env = TradingEnv(
        df=env_frame,
        window_size=args.ppo_window,
        initial_balance=args.initial_balance,
        spread=args.spread,
        slippage=args.slippage,
    )
    agent = PPOAgent(
        env=env,
        device=args.device,
        ppo_kwargs={"verbose": args.ppo_verbose, "seed": args.seed},
    )

    started = time.perf_counter()
    agent.train(total_timesteps=args.ppo_steps)
    elapsed = time.perf_counter() - started

    out_path = args.out_dir / f"ppo_{args.symbol.lower()}.zip"
    agent.save(out_path)
    logger.info("PPO saved -> %s (%.1fs)", out_path, elapsed)
    return {
        "artifact": str(out_path),
        "bars": len(env_frame),
        "timesteps": args.ppo_steps,
        "seconds": round(elapsed, 1),
    }


def _write_report(args: argparse.Namespace, payload: dict[str, object]) -> Path:
    report_path = args.report or (
        ROOT / "reports" / f"training_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    report_path = report_path if report_path.is_absolute() else (ROOT / report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    try:
        csv_path = _resolve_csv(args.data)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Loading %s", csv_path)
    try:
        frame = load_ohlcv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.error("Could not load %s: %s", csv_path, exc)
        return 1

    logger.info(
        "Loaded %d bars: %s -> %s",
        len(frame),
        frame.index.min(),
        frame.index.max(),
    )

    payload: dict[str, object] = {
        "data": str(csv_path),
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe.upper(),
        "bars": len(frame),
        "first_bar": str(frame.index.min()),
        "last_bar": str(frame.index.max()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "results": {},
    }
    results: dict[str, object] = {}

    targets = ["ppo", "lstm"] if args.model == "all" else [args.model]
    failures = 0

    for target in targets:
        try:
            if args.dry_run and target == "ppo":
                logger.info(
                    "[dry-run] PPO needs >= %d bars, %d available.", MIN_PPO_BARS, len(frame)
                )
                results[target] = {"dry_run": True, "bars": len(frame)}
                continue
            results[target] = (
                train_lstm(frame, args) if target == "lstm" else train_ppo(frame, args)
            )
        except (RuntimeError, ValueError, ImportError, OSError) as exc:
            failures += 1
            logger.error("%s training failed: %s", target.upper(), exc)
            results[target] = {"error": str(exc)}

    payload["results"] = results
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report_path = _write_report(args, payload)
    logger.info("Report -> %s", report_path)

    for result in results.values():
        artifact = result.get("artifact") if isinstance(result, dict) else None
        if not artifact:
            continue
        path = Path(str(artifact))
        try:
            shown = path.relative_to(ROOT).as_posix()
        except ValueError:
            shown = str(path)
        logger.info("Artifact: %s", path)
        print(f"MODEL_PATH={shown}")

    if failures and failures == len(targets):
        logger.error("All requested models failed. See the messages above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
