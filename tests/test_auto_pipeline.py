"""Unit tests for `scripts/auto_pipeline.py`.

The pipeline itself shells out to `fetch_history.py` / `train_models.py` (MT5 and
torch, both unavailable in CI), so these tests cover the decision logic and the
command/state plumbing with `run_command` patched out.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts import auto_pipeline as ap

CSV_HEADER = "time,open,high,low,close,tick_volume\n"


def _write_csv(path: Path, last_bar: datetime, rows: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    for offset in range(rows):
        stamp = last_bar - timedelta(minutes=5 * (rows - 1 - offset))
        lines.append(f"{stamp:%Y-%m-%d %H:%M:%S},1.0,1.1,0.9,1.05,10\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _touch_at(path: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(path, (timestamp, timestamp))


def _args(tmp_path: Path, **overrides) -> object:
    argv = [
        "--ensure",
        "--algo",
        "ppo",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M5",
        "--data-dir",
        str(tmp_path / "data"),
        "--model-dir",
        str(tmp_path / "models"),
        "--state-file",
        str(tmp_path / "reports" / "state.json"),
        "--report-dir",
        str(tmp_path / "reports"),
    ]
    return ap.parse_args(argv + list(overrides.get("extra", [])))


# --------------------------------------------------------------------------- #
# Configuration / artifact helpers
# --------------------------------------------------------------------------- #
def test_read_env_file_parses_values_and_ignores_noise(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nALGORITHM=ppo\nSYMBOL='XAUUSD'\n\nBROKEN_LINE\nTIMEFRAME = \"M15\"\n",
        encoding="utf-8",
    )
    values = ap.read_env_file(env)
    assert values["ALGORITHM"] == "ppo"
    assert values["SYMBOL"] == "XAUUSD"
    assert values["TIMEFRAME"] == "M15"
    assert "BROKEN_LINE" not in values


def test_read_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert ap.read_env_file(tmp_path / "nope.env") == {}


def test_env_value_prefers_process_environment(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("SYMBOL=EURUSD\n", encoding="utf-8")
    monkeypatch.setenv("SYMBOL", "GBPUSD")
    assert ap.env_value("SYMBOL", "XAUUSD", env) == "GBPUSD"
    monkeypatch.delenv("SYMBOL")
    assert ap.env_value("SYMBOL", "XAUUSD", env) == "EURUSD"
    assert ap.env_value("MISSING", "fallback", env) == "fallback"


@pytest.mark.parametrize(
    ("algo", "expected"),
    [
        ("ppo", ("ppo_xauusd.zip",)),
        ("LSTM", ("lstm_xauusd.pt",)),
        ("ensemble", ("ppo_xauusd.zip", "lstm_xauusd.pt")),
        ("transformer", ("transformer_xauusd.pt",)),
        ("unknown", ()),
    ],
)
def test_required_artifact_names(algo: str, expected: tuple[str, ...]) -> None:
    assert ap.required_artifact_names(algo, "XAUUSD") == expected


def test_last_bar_time_reads_tail_of_csv(tmp_path: Path) -> None:
    last_bar = datetime(2026, 1, 2, 15, 30)
    path = _write_csv(tmp_path / "XAUUSD_M5.csv", last_bar)
    assert ap.last_bar_time(path) == last_bar
    assert ap.count_rows(path) == 5


def test_last_bar_time_handles_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("open,close\n1,1\n", encoding="utf-8")
    assert ap.last_bar_time(path) is None
    assert ap.last_bar_time(tmp_path / "absent.csv") is None


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def test_build_plan_requests_fetch_when_no_data(tmp_path: Path) -> None:
    plan = ap.build_plan("ppo", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    assert plan.fetch_needed is True
    assert "no historical CSV" in plan.fetch_reason
    assert plan.missing_models == ["ppo_xauusd.zip"]
    assert plan.train_needed is True
    assert plan.models_to_train == "ppo"


def test_build_plan_reports_nothing_to_do_when_fresh(tmp_path: Path) -> None:
    now = datetime(2026, 1, 3, 12, 0)
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_csv = _write_csv(data_dir / "XAUUSD_M5.csv", now)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "lstm_xauusd.pt"
    model.write_bytes(b"weights")
    _touch_at(data_csv, now - timedelta(hours=2))
    _touch_at(model, now - timedelta(hours=1))

    plan = ap.build_plan("lstm", "XAUUSD", "M5", data_dir, model_dir, now=now)

    assert plan.fetch_needed is False
    assert plan.missing_models == []
    assert plan.stale_models == []
    assert plan.train_needed is False
    assert plan.models_to_train == ""
    assert "data_age=2.0h" in plan.summary()


def test_build_plan_flags_stale_data(tmp_path: Path) -> None:
    now = datetime(2026, 1, 5, 12, 0)
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", now)
    _touch_at(data_csv, now - timedelta(hours=30))

    plan = ap.build_plan(
        "ppo",
        "XAUUSD",
        "M5",
        tmp_path / "data",
        tmp_path / "models",
        now=now,
        max_data_age_hours=24,
    )

    assert plan.fetch_needed is True
    assert "30.0h old" in plan.fetch_reason


def test_build_plan_retrains_models_older_than_data(tmp_path: Path) -> None:
    now = datetime(2026, 1, 5, 12, 0)
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", now)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ppo = model_dir / "ppo_xauusd.zip"
    lstm = model_dir / "lstm_xauusd.pt"
    ppo.write_bytes(b"x")
    lstm.write_bytes(b"y")
    _touch_at(ppo, now - timedelta(days=3))
    _touch_at(lstm, now - timedelta(hours=1))
    _touch_at(data_csv, now - timedelta(hours=2))

    plan = ap.build_plan("ensemble", "XAUUSD", "M5", tmp_path / "data", model_dir, now=now)

    assert plan.stale_models == ["ppo_xauusd.zip"]
    assert plan.models_to_train == "ppo"
    assert any("predates the current dataset" in note for note in plan.notes)


def test_build_plan_retrains_on_age_limit(tmp_path: Path) -> None:
    now = datetime(2026, 3, 1, 12, 0)
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", now)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "lstm_xauusd.pt"
    model.write_bytes(b"y")
    # Data is older than the model (so "predates the dataset" cannot fire) but the
    # model itself is past the age limit.
    _touch_at(data_csv, now - timedelta(days=60))
    _touch_at(model, now - timedelta(days=45))

    plan = ap.build_plan(
        "lstm",
        "XAUUSD",
        "M5",
        tmp_path / "data",
        model_dir,
        now=now,
        max_data_age_hours=24 * 365,
        max_model_age_days=30,
    )

    assert plan.stale_models == ["lstm_xauusd.pt"]
    assert any("45d old" in note for note in plan.notes)


def test_build_plan_marks_untrainable_algorithms(tmp_path: Path) -> None:
    now = datetime(2026, 1, 5, 12, 0)
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", now)
    plan = ap.build_plan(
        "transformer", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models", now=now
    )

    assert plan.untrainable == ["transformer"]
    assert plan.missing_models == ["transformer_xauusd.pt"]
    assert plan.models_to_train == ""
    assert any("no trainer" in note for note in plan.notes)
    del data_csv


def test_build_plan_handles_unknown_algorithm(tmp_path: Path) -> None:
    plan = ap.build_plan("martingale", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    assert plan.missing_models == []
    assert plan.train_needed is False
    assert any("unknown algorithm" in note for note in plan.notes)


def test_models_to_train_maps_both_components(tmp_path: Path) -> None:
    plan = ap.build_plan("ensemble", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    assert set(plan.missing_models) == {"ppo_xauusd.zip", "lstm_xauusd.pt"}
    assert plan.models_to_train == "all"


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def test_fetch_command_uses_stable_output_and_incremental_start(tmp_path: Path) -> None:
    args = _args(tmp_path)
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", datetime(2026, 1, 2, 10, 0))

    cmd = ap.fetch_command(args, data_csv)

    assert "--out" in cmd
    assert cmd[cmd.index("--out") + 1] == str(data_csv)
    assert cmd[cmd.index("--start") + 1] == "2025-12-30"  # 3-day overlap before the last bar
    assert "--synthetic" not in cmd


def test_fetch_command_synthetic_has_no_start(tmp_path: Path) -> None:
    args = ap.parse_args(
        [
            "--ensure",
            "--synthetic",
            "--bars",
            "1234",
            "--symbol",
            "XAUUSD",
            "--timeframe",
            "M5",
            "--data-dir",
            str(tmp_path),
        ]
    )
    cmd = ap.fetch_command(args, None)
    assert "--synthetic" in cmd
    assert cmd[cmd.index("--bars") + 1] == "1234"
    assert "--start" not in cmd


def test_train_command_targets_report_and_output_dir(tmp_path: Path) -> None:
    args = _args(tmp_path)
    report = tmp_path / "reports" / "r.json"
    cmd = ap.train_command(args, tmp_path / "data" / "XAUUSD_M5.csv", "all", report)

    assert cmd[1].endswith("train_models.py")
    assert cmd[cmd.index("--model") + 1] == "all"
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path / "models")
    assert cmd[cmd.index("--report") + 1] == str(report)


def test_run_command_dry_run_does_not_execute(tmp_path: Path) -> None:
    marker = tmp_path / "should_not_exist"
    cmd = ["python", "-c", f"open({str(marker)!r}, 'w').close()"]
    assert ap.run_command(cmd, dry_run=True) == 0
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class _Recorder:
    """Stand-in for `run_command` that records invocations."""

    def __init__(self, codes: list[int] | None = None, report: dict | None = None) -> None:
        self.calls: list[list[str]] = []
        self.codes = list(codes or [])
        self.report = report

    def __call__(self, cmd: list[str], *, dry_run: bool = False) -> int:
        self.calls.append(cmd)
        if self.report is not None and "--report" in cmd:
            Path(cmd[cmd.index("--report") + 1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[cmd.index("--report") + 1]).write_text(
                json.dumps(self.report), encoding="utf-8"
            )
        return self.codes.pop(0) if self.codes else 0


def test_execute_plan_fetches_and_trains_then_writes_state(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    recorder = _Recorder(report={"results": {"lstm": {"features": 247, "val_accuracy": 0.41}}})
    monkeypatch.setattr(ap, "run_command", recorder)

    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", datetime(2026, 1, 2, 10, 0))
    plan = ap.build_plan("ppo", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    plan.fetch_needed = True
    plan.fetch_reason = "stale"
    plan.data_csv = data_csv
    plan.missing_models = ["ppo_xauusd.zip"]

    code = ap.execute_plan(
        args,
        plan,
        state_path=tmp_path / "reports" / "state.json",
        report_dir=tmp_path / "reports",
        dry_run=False,
    )

    assert code == 0
    assert len(recorder.calls) == 2
    assert recorder.calls[0][1].endswith("fetch_history.py")
    assert recorder.calls[1][1].endswith("train_models.py")
    state = json.loads((tmp_path / "reports" / "state.json").read_text(encoding="utf-8"))
    assert state["ok"] is True
    assert state["train"]["model"] == "ppo"
    assert state["train"]["features"] == 247
    assert state["train"]["lstm_val_accuracy"] == 0.41
    assert state["version"] == ap.STATE_VERSION


def test_execute_plan_stops_when_fetch_fails(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    recorder = _Recorder(codes=[1])
    monkeypatch.setattr(ap, "run_command", recorder)

    plan = ap.build_plan("ppo", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    plan.fetch_needed = True
    plan.fetch_reason = "no data"

    code = ap.execute_plan(
        args,
        plan,
        state_path=tmp_path / "reports" / "state.json",
        report_dir=tmp_path / "reports",
        dry_run=False,
    )

    assert code == 1
    assert len(recorder.calls) == 1  # no training attempted without data
    state = json.loads((tmp_path / "reports" / "state.json").read_text(encoding="utf-8"))
    assert state["ok"] is False


def test_execute_plan_reports_training_failure(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(ap, "run_command", _Recorder(codes=[2]))
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", datetime(2026, 1, 2, 10, 0))

    plan = ap.build_plan("lstm", "XAUUSD", "M5", tmp_path / "data", tmp_path / "models")
    plan.data_csv = data_csv
    plan.missing_models = ["lstm_xauusd.pt"]

    code = ap.execute_plan(
        args,
        plan,
        state_path=tmp_path / "reports" / "state.json",
        report_dir=tmp_path / "reports",
        dry_run=False,
    )

    assert code == 1
    state = json.loads((tmp_path / "reports" / "state.json").read_text(encoding="utf-8"))
    assert state["ok"] is False
    assert state["train"]["exit_code"] == 2


def test_run_once_status_never_fails_and_runs_nothing(tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(ap, "run_command", recorder)
    args = ap.parse_args(
        [
            "--status",
            "--symbol",
            "XAUUSD",
            "--timeframe",
            "M5",
            "--data-dir",
            str(tmp_path / "data"),
            "--model-dir",
            str(tmp_path / "models"),
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )
    assert ap.run_once(args) == 0
    assert recorder.calls == []
    assert not (tmp_path / "state.json").exists()


def test_run_once_skips_work_when_everything_is_fresh(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now()
    data_csv = _write_csv(tmp_path / "data" / "XAUUSD_M5.csv", now)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ppo = model_dir / "ppo_xauusd.zip"
    ppo.write_bytes(b"x")
    _touch_at(ppo, now)

    recorder = _Recorder()
    monkeypatch.setattr(ap, "run_command", recorder)
    args = ap.parse_args(
        [
            "--ensure",
            "--algo",
            "ppo",
            "--symbol",
            "XAUUSD",
            "--timeframe",
            "M5",
            "--data-dir",
            str(tmp_path / "data"),
            "--model-dir",
            str(model_dir),
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )
    assert ap.run_once(args) == 0
    assert recorder.calls == []
    assert data_csv.exists()  # dataset untouched, no fetch re-run


def test_loop_with_max_runs_exits_cleanly(tmp_path: Path, monkeypatch) -> None:
    ticks: list[int] = []

    def fake_run_once(_args):
        ticks.append(1)
        return 0

    monkeypatch.setattr(ap, "run_once", fake_run_once)
    monkeypatch.setattr(ap.time, "sleep", lambda _seconds: None)

    code = ap.main(
        [
            "--loop",
            "--max-runs",
            "3",
            "--tick-seconds",
            "0",
            "--data-dir",
            str(tmp_path / "data"),
            "--model-dir",
            str(tmp_path / "models"),
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )

    assert code == 0
    assert len(ticks) == 3


def test_summarise_report_flattens_nested_results() -> None:
    payload = {"results": {"lstm": {"features": 248, "val_accuracy": 0.52}, "ppo": {"error": "x"}}}
    summary = ap.summarise_report(payload)
    assert summary["features"] == 248
    assert summary["lstm_val_accuracy"] == 0.52
    assert ap.summarise_report({}) == {"features": None, "lstm_val_accuracy": None}


def test_parse_args_requires_a_mode() -> None:
    with pytest.raises(SystemExit):
        ap.parse_args([])
