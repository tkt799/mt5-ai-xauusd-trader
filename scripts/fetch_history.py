"""
MT5 AI/ML Trading Bot - Enterprise Edition
scripts/fetch_history.py
Download historical OHLCV bars into data/raw/ for backtesting and model training.

Data sources
------------
* Native MetaTrader 5 terminal (Windows)  -> requires MT5_* settings in .env
* MetaAPI cloud bridge                    -> requires METAAPI_TOKEN / METAAPI_ACCOUNT_ID
* Synthetic generator (--synthetic)       -> no credentials, no terminal, useful for smoke tests

Examples
--------
  # 2 years of XAUUSD M5 from the MT5 terminal (default)
  python scripts/fetch_history.py --symbol XAUUSD --timeframe M5 --start 2024-01-01 --end 2025-12-31

  # Incremental update: only the missing tail, merged into the existing CSV
  python scripts/fetch_history.py --start 2026-01-01

  # Offline smoke test without MT5
  python scripts/fetch_history.py --synthetic --bars 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("fetch_history")

REQUIRED_COLUMNS = ["open", "high", "low", "close"]
SYNTHETIC_FREQ = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV bars into data/raw/ (CSV + metadata JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol.")
    parser.add_argument(
        "--timeframe", default="M5", help="Chart timeframe (M1, M5, M15, H1, H4, D1)."
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD). Default: 730 days before --end.",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output CSV path. Default: data/raw/{SYMBOL}_{TF}_{start}_{end}.csv",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Split the request into chunks of N days (0 = single request).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic bars instead of contacting MT5/MetaAPI.",
    )
    parser.add_argument("--bars", type=int, default=20000, help="Number of synthetic bars.")
    parser.add_argument(
        "--regime",
        default="ranging",
        choices=[
            "trending",
            "ranging",
            "volatile",
            "whipsaw",
            "mean_reversion",
            "low_volatility_drift",
        ],
        help="Synthetic market regime.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synthetic generator seed.")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge with an existing output CSV (overwrite instead).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def _freq(timeframe: str) -> str:
    return SYNTHETIC_FREQ.get(timeframe.upper(), "5min")


def _chunks(start: datetime, end: datetime, chunk_days: int) -> Iterator[tuple[datetime, datetime]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end]."""
    if chunk_days <= 0:
        yield start, end
        return

    cursor = start
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, enforce a UTC-naive time column and drop bad rows."""
    frame = df.copy()
    if not isinstance(frame.index, pd.RangeIndex) and "time" not in frame.columns:
        frame.index.name = "time"
        frame = frame.reset_index()

    frame.columns = [str(col).lower().strip() for col in frame.columns]
    if "time" not in frame.columns:
        raise ValueError("DataFrame has no 'time' column or DatetimeIndex.")

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])

    for column in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = frame.get("real_volume", 0)

    frame = frame.dropna(subset=[column for column in REQUIRED_COLUMNS if column in frame.columns])
    frame = frame[frame["close"] > 0]
    frame = frame.drop_duplicates(subset=["time"], keep="last")
    return frame.sort_values("time").reset_index(drop=True)


def fetch_from_terminal(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> pd.DataFrame:
    """Fetch bars through MT5Connector (native terminal or MetaAPI fallback)."""
    from src.core.config import get_config
    from src.core.exceptions import MT5ConnectionError, MT5DataError
    from src.trading.mt5_connector import MT5Connector

    try:
        cfg = get_config()
    except Exception as exc:  # pydantic ValidationError when .env is missing/incomplete
        raise RuntimeError(
            "Could not load configuration from .env. Run start.bat -> [2] setup wizard "
            f"(--setup) first. Details: {exc}"
        ) from exc

    connector = MT5Connector(cfg)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    windows = list(_chunks(start, end, chunk_days))
    try:
        connector.connect()
        source = "MetaAPI" if connector.use_metaapi else "MT5 terminal"
        logger.info("Connected via %s (%d chunk(s) to fetch)", source, len(windows))
        for index, (chunk_start, chunk_end) in enumerate(windows, start=1):
            try:
                chunk = connector.get_rates_range(symbol, timeframe, chunk_start, chunk_end)
            except (MT5DataError, MT5ConnectionError) as exc:
                failures.append(f"{chunk_start:%Y-%m-%d}->{chunk_end:%Y-%m-%d}: {exc}")
                logger.warning("Chunk %d failed (%s -> %s): %s", index, chunk_start, chunk_end, exc)
                continue

            if chunk is None or chunk.empty:
                logger.debug("Chunk %d returned no bars (%s -> %s)", index, chunk_start, chunk_end)
                continue

            frames.append(chunk)
            logger.info(
                "Chunk %d/%d: %s -> %s, %d bars",
                index,
                len(windows),
                chunk_start.date(),
                chunk_end.date(),
                len(chunk),
            )
    finally:
        connector.disconnect()

    if failures:
        logger.warning("%d chunk(s) failed; re-run with --start/--end to backfill.", len(failures))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_synthetic(
    symbol: str,
    timeframe: str,
    bars: int,
    regime: str,
    seed: int,
    end: datetime,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars so the pipeline can run without MT5."""
    from src.utils.synthetic_data import ScenarioGenerator

    # Walk backwards so the last synthetic bar lands on --end.
    step = pd.Timedelta(_freq(timeframe))
    start = (end - step * bars).replace(microsecond=0)
    generator = ScenarioGenerator(seed=seed)
    frame = generator.generate(
        n_steps=bars,
        regime=regime,
        start_price=2300.0,
        start_date=start,
        freq=_freq(timeframe),
    )
    return frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    else:
        end = datetime.now()
    start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else end - timedelta(days=730)

    if args.synthetic:
        start = (end - pd.Timedelta(_freq(args.timeframe)) * args.bars).replace(microsecond=0)

    if start >= end:
        logger.error("--start (%s) must be before --end (%s).", start, end)
        return 2

    out_path = args.out or (
        ROOT
        / "data"
        / "raw"
        / f"{args.symbol.upper()}_{args.timeframe.upper()}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    )
    out_path = out_path if out_path.is_absolute() else (ROOT / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    if args.synthetic:
        logger.info(
            "Generating %d synthetic %s bars (regime=%s, seed=%d).",
            args.bars,
            args.timeframe,
            args.regime,
            args.seed,
        )
        raw = build_synthetic(args.symbol, args.timeframe, args.bars, args.regime, args.seed, end)
        source = f"synthetic:{args.regime}"
    else:
        logger.info(
            "Fetching %s %s from %s to %s (chunk=%d days).",
            args.symbol,
            args.timeframe,
            start.date(),
            end.date(),
            args.chunk_days,
        )
        try:
            raw = fetch_from_terminal(args.symbol, args.timeframe, start, end, args.chunk_days)
        except Exception as exc:  # report the reason instead of a traceback
            logger.error(
                "Could not fetch history from MT5/MetaAPI (%s: %s). "
                "Check that the terminal is running and logged in, or try --synthetic.",
                type(exc).__name__,
                exc,
            )
            return 1
        source = "mt5"

    if raw.empty:
        logger.error(
            "No bars returned. Check that MT5 is running and logged in, that the symbol name "
            "matches your broker (%s), and that the history is downloaded in the terminal "
            "(Tools -> Options -> Charts -> Max bars in chart).",
            args.symbol,
        )
        return 1

    frame = _normalise(raw)
    frame = frame[(frame["time"] >= start) & (frame["time"] <= end)]

    if frame.empty:
        logger.error("All fetched bars fell outside the requested range.")
        return 1

    if not args.no_merge and out_path.exists():
        try:
            previous = pd.read_csv(out_path)
            frame = _normalise(pd.concat([previous, frame], ignore_index=True))
            logger.info("Merged with existing %s (%d bars total).", out_path, len(frame))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logger.warning("Could not merge existing CSV (%s); overwriting.", exc)

    frame.to_csv(out_path, index=False)
    meta = {
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe.upper(),
        "source": source,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "rows": len(frame),
        "first_bar": frame["time"].iloc[0].isoformat(),
        "last_bar": frame["time"].iloc[-1].isoformat(),
        "out_path": str(out_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    logger.info("Saved %d bars -> %s", len(frame), out_path)
    logger.info("Metadata -> %s", meta_path)
    logger.info(
        'Next: python scripts/train_models.py --data "%s" --model all',
        out_path,
    )
    if frame["time"].iloc[-1] < end - timedelta(days=3):
        logger.warning(
            "Last bar is %s (older than requested end %s). The broker may not have history "
            "that recent; open an XAUUSD chart in MT5 and scroll back to force a download.",
            frame["time"].iloc[-1],
            end,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
