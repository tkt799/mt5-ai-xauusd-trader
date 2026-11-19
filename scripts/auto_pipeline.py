"""
MT5 AI/ML Trading Bot - Enterprise Edition
scripts/auto_pipeline.py
Self-maintaining data + model pipeline: detect missing/stale artifacts, fetch bars, retrain.

Why
---
``main.py`` only *checks whether a checkpoint file exists*. When it does not exist
(``--algo ppo|lstm``) the wrapper is built from random weights or with no estimator
at all, so the bot happily runs and never emits a trade. Nothing in the repository
refreshed ``data/raw`` or retrained the weights, so keeping the AI current was a
manual chore.

This script closes that gap:

* ``--status``  show what is missing / stale (no side effects).
* ``--ensure``  fetch new bars when the data is old, train the missing or stale
  models, then write a state file. Safe to run from a scheduler or ``start.bat``.
* ``--loop``    long-running supervisor: re-evaluates the plan on a timer, so the
  AI keeps learning from fresh market data without an operator.

Examples
--------
  # What would the system do?
  python scripts/auto_pipeline.py --status

  # Fetch the missing tail + train whatever is missing, once
  python scripts/auto_pipeline.py --ensure

  # Keep data < 12h old and retrain at most weekly (Ctrl+C to stop)
  python scripts/auto_pipeline.py --loop --refresh-hours 12 --retrain-days 7

  # Offline smoke test: synthetic bars, no MT5 credentials needed
  python scripts/auto_pipeline.py --ensure --synthetic --bars 20000

Notes
-----
* The incremental fetch merges into ``data/raw/<SYMBOL>_<TF>.csv`` so daily runs do
  not litter the folder with one CSV per day.
* ``transformer`` / ``dreamer`` have no trainer in this repository; the script says
  so instead of pretending to train them.
* Training needs PyTorch (``pip install -r requirements.txt``); the guard message
  points at ``start.bat [8]`` when it is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("auto_pipeline")

DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "M5"
DEFAULT_ALGO = "ensemble"

#: Artifact file names per algorithm (``{symbol}`` is lower-cased).
MODEL_FILES: dict[str, tuple[str, ...]] = {
    "ppo": ("ppo_{symbol}.zip",),
    "lstm": ("lstm_{symbol}.pt",),
    "ensemble": ("ppo_{symbol}.zip", "lstm_{symbol}.pt"),
    "transformer": ("transformer_{symbol}.pt",),
    "dreamer": ("dreamer_{symbol}.pt",),
}

#: Algorithms this repository can actually train (``scripts/train_models.py``).
TRAINABLE = {"ppo", "lstm", "ensemble"}

STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# Configuration helpers (work with or without python-dotenv)
# --------------------------------------------------------------------------- #
def read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file without importing python-dotenv.

    Args:
        path: Path to the ``.env`` file.

    Returns:
        Mapping of key -> value (empty when the file does not exist).
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        values[key.strip().upper()] = value
    return values


def env_value(key: str, default: str, env_file: Path) -> str:
    """Resolve ``key`` from the process environment first, then ``.env``."""
    from_process = os.environ.get(key)
    if from_process:
        return from_process
    return read_env_file(env_file).get(key.upper(), default) or default


# --------------------------------------------------------------------------- #
# Artifact discovery
# --------------------------------------------------------------------------- #
def resolve_targets(args: argparse.Namespace, env_file: Path | None = None) -> argparse.Namespace:
    """Fill algorithm/symbol/timeframe from ``.env`` when the CLI left them empty."""
    env_file = env_file or (ROOT / ".env")
    args.algorithm = (args.algorithm or env_value("ALGORITHM", DEFAULT_ALGO, env_file)).lower()
    args.symbol = (args.symbol or env_value("SYMBOL", DEFAULT_SYMBOL, env_file)).upper()
    args.timeframe = (args.timeframe or env_value("TIMEFRAME", DEFAULT_TIMEFRAME, env_file)).upper()
    return args


def required_artifact_names(algo: str, symbol: str) -> tuple[str, ...]:
    """Return the checkpoint file names ``main.py`` expects for ``algo``."""
    templates = MODEL_FILES.get(algo.lower(), ())
    return tuple(template.format(symbol=symbol.lower()) for template in templates)


def latest_data_csv(data_dir: Path) -> Path | None:
    """Return the newest CSV in ``data_dir`` (meta/sidecar files excluded)."""
    if not data_dir.exists():
        return None
    candidates = [
        path
        for path in data_dir.glob("*.csv")
        if path.is_file() and not path.name.endswith((".meta.csv",))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def last_bar_time(csv_path: Path) -> datetime | None:
    """Read the newest timestamp from a CSV without parsing the whole file."""
    try:
        with csv_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    header_line = None
    with csv_path.open("rb") as handle:
        header_line = handle.readline().decode("utf-8", errors="replace")
    if not header_line:
        return None

    columns = [name.strip().lower() for name in header_line.split(",")]
    if "time" not in columns:
        return None
    index = columns.index("time")

    for line in reversed(tail.splitlines()):
        parts = line.split(",")
        if len(parts) <= index:
            continue
        stamp = parts[index].strip().strip('"')
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            continue
    return None


def count_rows(csv_path: Path) -> int:
    """Count data rows in a CSV (cheap enough for a few hundred thousand bars)."""
    try:
        with csv_path.open("rb") as handle:
            total = sum(1 for _ in handle)
    except OSError:
        return 0
    return max(0, total - 1)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    """What the pipeline intends to do on the next tick."""

    algorithm: str
    symbol: str
    timeframe: str
    data_csv: Path | None = None
    data_age_hours: float | None = None
    fetch_needed: bool = False
    fetch_reason: str = ""
    missing_models: list[str] = field(default_factory=list)
    stale_models: list[str] = field(default_factory=list)
    untrainable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def train_needed(self) -> bool:
        """True when at least one trainable model must be (re)built."""
        return bool(self.missing_models or self.stale_models)

    @property
    def models_to_train(self) -> str:
        """Map the required artifacts to ``train_models.py --model`` values."""
        wanted = set(self.missing_models) | set(self.stale_models)
        names = {name.split("_")[0] for name in wanted}
        if {"ppo", "lstm"} <= names:
            return "all"
        if "ppo" in names:
            return "ppo"
        if "lstm" in names:
            return "lstm"
        return ""

    def summary(self) -> str:
        """Human readable one-liner used by ``--status``."""
        parts = [f"algo={self.algorithm}", f"symbol={self.symbol}", f"tf={self.timeframe}"]
        if self.data_csv:
            parts.append(f"data={self.data_csv.name}")
            if self.data_age_hours is not None:
                parts.append(f"data_age={self.data_age_hours:.1f}h")
        else:
            parts.append("data=<none>")
        parts.append(f"fetch={'yes' if self.fetch_needed else 'no'}")
        parts.append(f"missing={self.missing_models or '-'}")
        parts.append(f"stale={self.stale_models or '-'}")
        return " | ".join(parts)


def build_plan(
    algorithm: str,
    symbol: str,
    timeframe: str,
    data_dir: Path,
    model_dir: Path,
    *,
    now: datetime | None = None,
    max_data_age_hours: float = 24.0,
    max_model_age_days: float = 30.0,
) -> Plan:
    """Decide what needs to happen for ``algorithm``.

    Args:
        algorithm: One of ppo/lstm/ensemble/transformer/dreamer.
        symbol: Trading symbol (case insensitive).
        timeframe: Base timeframe label, used in the data file name.
        data_dir: Folder that holds the raw CSVs.
        model_dir: Folder that holds the checkpoints ``main.py`` loads.
        now: Injectable clock (tests).
        max_data_age_hours: Refresh bars once the newest CSV is older than this.
        max_model_age_days: Retrain even without new data after this many days.

    Returns:
        A :class:`Plan` describing the next actions.
    """
    now = now or datetime.now()
    plan = Plan(algorithm=algorithm.lower(), symbol=symbol, timeframe=timeframe)
    symbol_lower = symbol.lower()

    # Prefer the stable file name produced by this pipeline, fall back to "newest".
    preferred = data_dir / f"{symbol.upper()}_{timeframe.upper()}.csv"
    plan.data_csv = preferred if preferred.exists() else latest_data_csv(data_dir)

    if plan.data_csv is None:
        plan.fetch_needed = True
        plan.fetch_reason = "no historical CSV in data/raw"
    else:
        age = now - datetime.fromtimestamp(plan.data_csv.stat().st_mtime)
        plan.data_age_hours = age.total_seconds() / 3600.0
        if plan.data_age_hours > max_data_age_hours:
            plan.fetch_needed = True
            plan.fetch_reason = (
                f"newest CSV is {plan.data_age_hours:.1f}h old (limit {max_data_age_hours:.1f}h)"
            )

    names = required_artifact_names(plan.algorithm, symbol_lower)
    if not names:
        plan.notes.append(f"unknown algorithm '{algorithm}' - nothing to check")
        return plan

    if plan.algorithm in ("transformer", "dreamer"):
        plan.untrainable.append(plan.algorithm)
        plan.notes.append(
            f"'{plan.algorithm}' has no trainer in this repository; "
            "train ppo/lstm and use --algo ensemble or ppo instead"
        )

    data_mtime = plan.data_csv.stat().st_mtime if plan.data_csv else 0.0
    for name in names:
        path = model_dir / name
        if not path.exists():
            plan.missing_models.append(name)
            continue
        model_age_days = (now - datetime.fromtimestamp(path.stat().st_mtime)).days
        if path.stat().st_mtime < data_mtime:
            plan.stale_models.append(name)
            plan.notes.append(f"{name} predates the current dataset - retraining")
        elif model_age_days > max_model_age_days:
            plan.stale_models.append(name)
            plan.notes.append(
                f"{name} is {model_age_days}d old (limit {max_model_age_days:.0f}d) - retraining"
            )
    return plan


# --------------------------------------------------------------------------- #
# Command construction / execution
# --------------------------------------------------------------------------- #
def fetch_command(args: argparse.Namespace, data_csv: Path | None) -> list[str]:
    """Build the ``fetch_history.py`` invocation for the current settings."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_history.py"),
        "--symbol",
        args.symbol,
        "--timeframe",
        args.timeframe,
        "--out",
        str(
            data_csv
            or (Path(args.data_dir) / f"{args.symbol.upper()}_{args.timeframe.upper()}.csv")
        ),
    ]
    if args.synthetic:
        cmd += ["--synthetic", "--bars", str(args.bars)]
    elif data_csv is not None:
        # Incremental refresh: re-download a small overlap so the merge is seamless.
        last_bar = last_bar_time(data_csv)
        if last_bar is not None:
            start = (last_bar - timedelta(days=args.overlap_days)).date().isoformat()
            cmd += ["--start", start]
    if args.no_merge:
        cmd.append("--no-merge")
    return cmd


def train_command(args: argparse.Namespace, data_csv: Path, model: str, report: Path) -> list[str]:
    """Build the ``train_models.py`` invocation for ``model`` (ppo/lstm/all)."""
    return [
        sys.executable,
        str(ROOT / "scripts" / "train_models.py"),
        "--data",
        str(data_csv),
        "--symbol",
        args.symbol,
        "--timeframe",
        args.timeframe,
        "--model",
        model,
        "--out-dir",
        str(Path(args.model_dir)),
        "--report",
        str(report),
    ]


def run_command(cmd: list[str], *, dry_run: bool = False) -> int:
    """Execute ``cmd`` (or print it) and return the exit code."""
    printable = " ".join(cmd)
    if dry_run:
        logger.info("[dry-run] %s", printable)
        return 0
    logger.info("$ %s", printable)
    try:
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    except OSError as exc:
        logger.error("Could not start %s: %s", cmd[1] if len(cmd) > 1 else cmd[0], exc)
        return 1
    return completed.returncode


def read_training_report(path: Path) -> dict:
    """Best-effort read of a ``--report`` JSON so the state file can summarise it."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarise_report(payload: dict) -> dict:
    """Flatten ``train_models.py --report`` output into a few state-file fields.

    The report nests per-model results under ``results``::

        {"results": {"lstm": {"features": 248, "val_accuracy": 0.41}}}

    Args:
        payload: Parsed report JSON.

    Returns:
        ``{"features": int | None, "lstm_val_accuracy": float | None}``.
    """
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    features = None
    for entry in results.values():
        if isinstance(entry, dict) and entry.get("features"):
            features = entry["features"]
            break
    lstm = results.get("lstm") if isinstance(results.get("lstm"), dict) else {}
    return {"features": features, "lstm_val_accuracy": lstm.get("val_accuracy")}


def write_state(state_path: Path, payload: dict) -> None:
    """Persist the last pipeline outcome atomically."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": STATE_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(state_path)


# --------------------------------------------------------------------------- #
# Core actions
# --------------------------------------------------------------------------- #
def execute_plan(
    args: argparse.Namespace,
    plan: Plan,
    *,
    state_path: Path,
    report_dir: Path,
    dry_run: bool,
) -> int:
    """Fetch data and/or train models as described by ``plan``.

    Returns:
        Process exit code: 0 = everything the plan asked for succeeded.
    """
    result: dict[str, object] = {
        "algorithm": plan.algorithm,
        "symbol": plan.symbol,
        "timeframe": plan.timeframe,
        "fetch": None,
        "train": None,
        "notes": plan.notes,
    }

    if plan.fetch_needed:
        data_csv = plan.data_csv or (
            Path(args.data_dir) / f"{args.symbol.upper()}_{args.timeframe.upper()}.csv"
        )
        code = run_command(fetch_command(args, plan.data_csv), dry_run=dry_run)
        result["fetch"] = {"exit_code": code, "reason": plan.fetch_reason}
        if code != 0:
            logger.error("Data fetch failed (exit %d): %s", code, plan.fetch_reason)
            write_state(state_path, {**result, "ok": False})
            return 1
        plan.data_csv = data_csv
        if not dry_run:
            plan.data_csv = latest_data_csv(Path(args.data_dir)) or plan.data_csv

    if plan.train_needed:
        if not plan.models_to_train:
            logger.warning("Nothing trainable among %s", plan.stale_models or plan.missing_models)
        elif plan.data_csv is None:
            logger.error("Cannot train: no dataset available in %s", args.data_dir)
            write_state(state_path, {**result, "ok": False})
            return 1
        else:
            report_dir.mkdir(parents=True, exist_ok=True)
            report = report_dir / f"training_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            code = run_command(
                train_command(args, plan.data_csv, plan.models_to_train, report),
                dry_run=dry_run,
            )
            summary = summarise_report(read_training_report(report)) if not dry_run else {}
            result["train"] = {
                "exit_code": code,
                "model": plan.models_to_train,
                "missing": plan.missing_models,
                "stale": plan.stale_models,
                "report": str(report),
                "features": summary.get("features"),
                "lstm_val_accuracy": summary.get("lstm_val_accuracy"),
            }
            if code != 0:
                logger.error("Training failed (exit %d)", code)
                write_state(state_path, {**result, "ok": False})
                return 1

    if plan.untrainable:
        result["untrainable"] = plan.untrainable

    write_state(state_path, {**result, "ok": True})
    return 0


def run_once(args: argparse.Namespace) -> int:
    """Evaluate the plan once and act on it."""
    resolve_targets(args)
    plan = build_plan(
        args.algorithm,
        args.symbol,
        args.timeframe,
        Path(args.data_dir),
        Path(args.model_dir),
        max_data_age_hours=args.refresh_hours,
        max_model_age_days=args.retrain_days,
    )
    logger.info("plan: %s", plan.summary())
    for note in plan.notes:
        logger.info("note: %s", note)

    if args.status:
        if not plan.fetch_needed and not plan.train_needed:
            logger.info("Nothing to do - data and models are up to date.")
        if plan.missing_models:
            logger.warning("Missing checkpoints: %s", ", ".join(plan.missing_models))
        if plan.stale_models:
            logger.warning("Stale checkpoints: %s", ", ".join(plan.stale_models))
        if plan.fetch_needed:
            logger.warning("Data refresh needed: %s", plan.fetch_reason)
        # A status run is informational: report problems, never fail the caller.
        return 0

    if not plan.fetch_needed and not plan.train_needed:
        logger.info("Nothing to do - data and models are up to date.")
        return 0

    return execute_plan(
        args,
        plan,
        state_path=Path(args.state_file),
        report_dir=Path(args.report_dir),
        dry_run=args.dry_run,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep market data and AI checkpoints up to date (auto-detect, fetch, train).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="Report what is missing/stale.")
    mode.add_argument("--ensure", action="store_true", help="Fetch/train as needed, then exit.")
    mode.add_argument("--loop", action="store_true", help="Run --ensure on a timer forever.")

    parser.add_argument(
        "--algo", dest="algorithm", help="ppo|lstm|ensemble (default: ALGORITHM in .env)"
    )
    parser.add_argument("--symbol", help="Trading symbol (default: SYMBOL in .env)")
    parser.add_argument("--timeframe", help="Base timeframe (default: TIMEFRAME in .env)")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"), help="Raw CSV folder.")
    parser.add_argument(
        "--model-dir", default=str(ROOT / "models" / "trained"), help="Checkpoint folder."
    )
    parser.add_argument(
        "--state-file",
        default=str(ROOT / "reports" / "auto_pipeline_state.json"),
        help="Where the last run outcome is recorded.",
    )
    parser.add_argument(
        "--report-dir", default=str(ROOT / "reports"), help="Training report folder."
    )
    parser.add_argument(
        "--refresh-hours", type=float, default=24.0, help="Refresh bars once the CSV is older."
    )
    parser.add_argument(
        "--retrain-days",
        type=float,
        default=30.0,
        help="Retrain even without new data after N days.",
    )
    parser.add_argument("--tick-seconds", type=int, default=900, help="Sleep between --loop ticks.")
    parser.add_argument(
        "--max-runs", type=int, default=0, help="Stop --loop after N ticks (0 = forever)."
    )
    parser.add_argument(
        "--overlap-days",
        type=float,
        default=3.0,
        help="How many days to re-download before the last stored bar.",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic bars (no MT5 needed)."
    )
    parser.add_argument("--bars", type=int, default=20000, help="Synthetic bar count.")
    parser.add_argument("--no-merge", action="store_true", help="Overwrite instead of merging.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    resolve_targets(args)

    if not args.loop:
        return run_once(args)

    ticks = 0
    exit_code = 0
    logger.info(
        "auto pipeline loop started (tick=%ss, refresh=%sh, retrain=%sd). Ctrl+C to stop.",
        args.tick_seconds,
        args.refresh_hours,
        args.retrain_days,
    )
    while True:
        ticks += 1
        try:
            exit_code = run_once(args)
        except KeyboardInterrupt:
            logger.info("Interrupted - stopping loop.")
            break
        except Exception as exc:  # the supervisor must survive anything
            logger.exception("Tick failed: %s", exc)
            exit_code = 1
        if args.max_runs and ticks >= args.max_runs:
            break
        try:
            time.sleep(args.tick_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted while sleeping - stopping loop.")
            break
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
