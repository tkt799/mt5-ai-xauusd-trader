# Automatic data + model refresh (`scripts/auto_pipeline.py`)

The bot needs two artifacts that it **cannot** create by itself at runtime:

| Artifact | Produced by | Consumed by |
|---|---|---|
| `data/raw/<SYMBOL>_<TF>.csv` | `scripts/fetch_history.py` | `scripts/train_models.py`, backtests |
| `models/trained/ppo_<symbol>.zip`, `lstm_<symbol>.pt` | `scripts/train_models.py` | `main.py` (`--algo ppo|lstm|ensemble`) |

`main.py` only checks whether those checkpoint files **exist**. Before this script the
missing step was always manual, and the failure mode was silent:

* `--algo ppo` with no `.zip` → `PPOAgent` has no estimator → every signal is `HOLD/0.0`;
* `--algo lstm` with no `.pt` → the network keeps its **random** initialisation and still
  returns BUY/SELL with a confidence figure;
* `--algo ensemble` with no files → the health gate fails with "no base models loaded".

`auto_pipeline.py` closes the loop: it detects what is missing or stale, fetches the new
bars, retrains, and records the outcome. Run it once, from a scheduler, or as a
long-running supervisor.

## Commands

```bash
# 1. What would happen? (read-only, always exit 0)
python scripts/auto_pipeline.py --status

# 2. Do it once: fetch the missing tail, train whatever is missing/stale
python scripts/auto_pipeline.py --ensure

# 3. Supervisor: keep data < 12h old, retrain at most weekly
python scripts/auto_pipeline.py --loop --refresh-hours 12 --retrain-days 7

# Offline smoke test (no MT5, no .env)
python scripts/auto_pipeline.py --ensure --synthetic --bars 20000
```

Windows: `start.bat auto` (menu entry **[10]**) runs `--ensure`.
Makefile: `make auto` / `make auto-status`.

## Decision rules

`build_plan()` decides on every tick:

1. **Data** — the newest CSV in `data/raw` (preferring `data/raw/<SYMBOL>_<TF>.csv`) must be
   younger than `--refresh-hours` (default 24). Otherwise `fetch_history.py` runs with
   `--start <last bar − 3 days>` and merges into the **same** file, so daily refreshes do not
   litter the folder with one CSV per day.
2. **Model** — for the configured algorithm (`ALGORITHM` in `.env`, or `--algo`):
   * file absent → train it;
   * file older than the dataset → train it (new bars arrived);
   * file older than `--retrain-days` (default 30) → train it anyway (drift insurance).
3. **Unsupported algorithms** — `transformer` and `dreamer` have no trainer in this
   repository; the script reports that instead of pretending.

Every run writes `reports/auto_pipeline_state.json`:

```json
{
  "version": 1,
  "updated_at": "2026-11-19T04:52:17",
  "algorithm": "lstm",
  "fetch": {"exit_code": 0, "reason": "no historical CSV in data/raw"},
  "train": {"exit_code": 1, "model": "lstm", "features": null},
  "ok": false
}
```

`ok: false` plus an `exit_code != 0` is what a scheduler should alert on.

## Scheduling

Windows Task Scheduler (daily 06:00, trading days):

```bat
schtasks /Create /TN "MT5 AI refresh" /SC DAILY /ST 06:00 ^
  /TR "cmd /c cd /d C:\path\to\mt5-ai-xauusd-trader && start.bat auto"
```

cron (every 6 hours):

```cron
0 */6 * * * cd /opt/mt5-ai-xauusd-trader && venv/bin/python scripts/auto_pipeline.py --ensure >> logs/auto_pipeline.log 2>&1
```

Long-running alternative (no scheduler needed, keeps retrying every `--tick-seconds`):

```bash
python scripts/auto_pipeline.py --loop --tick-seconds 900 --refresh-hours 12 --retrain-days 7
```

## What it does *not* do (yet)

* **No hot swap.** `main.py` builds the model once at startup; a checkpoint written while
  the bot is running is picked up on the next restart. Until a reload hook exists, restart
  the bot after a retrain (or run the pipeline on a schedule that ends before the session
  starts).
* **No model selection.** It refreshes the weights for the algorithm you configured; it does
  not compare PPO vs LSTM vs ensemble and pick the winner. Use `run_backtest` for that.
* **No `MODEL_PATH` gate bypass.** Startup still refuses to trade without a usable model —
  that is intentional (see `src/core/health.py::check_models`).
