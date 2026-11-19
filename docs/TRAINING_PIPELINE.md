# Data Acquisition & Model Training

How to get data into the bot, train the PPO/LSTM models, and automate the pipeline.

---

## 1. Where does "the database" come from?

There is nothing to download — the database is **created locally on first start**.

| Layer | Where it lives | How it is created |
| :--- | :--- | :--- |
| Trade / audit log DB | `DATABASE_URL` in `.env`, default `sqlite:///data/trades.db` | `src/core/database.py::get_engine()` creates the parent folder, the file (mode `0o600`), and enables `foreign_keys`, `WAL`, `busy_timeout`, `secure_delete`. Tables are created by `TradeLogger` / `AuditLogger` on startup. |
| Redis | `REDIS_URL` (optional) | Only needed for caching/queuing features; the bot runs without it. |
| PostgreSQL | `DATABASE_URL=postgresql://...` | Optional. Requires a running server; note `main.py` currently routes trade/audit logging to local SQLite unless the URL contains `sqlite`. Leave the default SQLite for demo/backtest. |

Startup validation (`src/core/config_validator.py::_check_placeholder_secrets`) treats the
template value `postgresql://trader:password@localhost:5432/mt5_trades` and any URL containing
`PASSWORD` / `SECRET` / `DUMMY` / `FAKE` / `CHANGE_ME` as a **critical** error and refuses to
start. On Windows, `start.bat` rewrites the value to `sqlite:///data/trades.db` automatically.

The database stores *trades and audit events*, **not** market history. Price history comes from
the sources in the next section.

---

## 2. Where does historical market data come from?

| Source | Requirements | Command |
| :--- | :--- | :--- |
| **MT5 terminal** (recommended) | Windows + MetaTrader 5 installed and logged in; `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` in `.env` | `python scripts/fetch_history.py --symbol XAUUSD --timeframe M5 --start 2023-01-01` |
| **MetaAPI cloud** | `METAAPI_TOKEN` + `METAAPI_ACCOUNT_ID` in `.env` | same command — `MT5Connector` switches automatically |
| **Synthetic** | nothing (no terminal, no credentials) | `python scripts/fetch_history.py --synthetic --bars 20000` |
| **Your own CSV** | `time,open,high,low,close,tick_volume` (+ optional `spread`) | pass it straight to `scripts/train_models.py --data path.csv` |

Notes:

- Bars are requested in monthly chunks (`--chunk-days`, default 30) so long ranges do not time
  out; failed chunks are logged and the rest still succeeds. Re-running with the same `--out`
  **merges** and de-duplicates by timestamp, so incremental updates are just a narrower `--start`.
- The connector returns **UTC naive** timestamps from MT5, which is what the feature pipeline
  expects.
- If the tail of the data is stale, open an `XAUUSD` chart in MT5 and scroll back once — the
  terminal downloads history lazily.
- Broker symbol names vary (`XAUUSD`, `XAUUSD.m`, `GOLD`); use the exact name from Market Watch.

Output: `data/raw/{SYMBOL}_{TF}_{start}_{end}.csv` plus a `*.meta.json` sidecar (rows, first/last
bar, source, generated_at). `data/` is git-ignored, so nothing is committed.

---

## 3. Train the models

```bash
# PPO (200k steps) + LSTM (10 epochs) on the newest CSV in data/raw
python scripts/train_models.py --data latest --model all

# LSTM only, quicker
python scripts/train_models.py --data latest --model lstm --lstm-epochs 3 --max-samples 5000

# Validate the data + feature pipeline without training
python scripts/train_models.py --data latest --dry-run
```

Windows one-click wrapper (runs fetch → train, keeps the window open):

```bat
train.bat              :: full pipeline (edit SYMBOL / TF / START at the top of the file)
train.bat synthetic    :: synthetic data, small run — smoke test without MT5
train.bat quick        :: dry-run validation only
```

Make targets: `make fetch`, `make train`, `make train-quick`.

### What the LSTM sees

1. `FeatureEngineer` builds the multi-timeframe indicator matrix (140+ columns with TA-Lib;
   ~13 columns without it, see troubleshooting).
2. Labels are 3-class from the forward log return over `--horizon` bars, split by
   `--label-quantile`: `BUY` (top quantile), `SELL` (bottom quantile), otherwise `HOLD`.
3. Windows of `--window` bars are sliced lazily (`torch Dataset`), so memory stays flat on
   multi-million-bar CSVs; `--max-samples` subsamples if you want faster epochs.
4. Chronological `--train-split` (default 0.8) — the validation slice is always the most recent
   bars, never shuffled across the split.

### What the PPO agent sees

Raw `open/high/low/close/volume` bars in `TradingEnv` (discrete 3 actions, spread/slippage
costs), which is the environment documented in `src/trading/trading_env.py`.

### Artifacts

| File | Loaded by |
| :--- | :--- |
| `models/trained/ppo_xauusd.zip` | `PPOAgent` / `EnsembleModel.load_ppo()` — `main.py` looks for `--model-dir/ppo_xauusd.zip` |
| `models/trained/lstm_xauusd.pt` | `LSTMModel` / `EnsembleModel.load_lstm()` — `--model-dir/lstm_xauusd.pt` |

`--model-dir` defaults to `models/trained`, and the filenames are fixed
(`ppo_{symbol}.zip`, `lstm_{symbol}.pt` — symbol lower-cased). A JSON report is written to
`reports/training_report_<timestamp>.json` (rows, feature count, label distribution, val
accuracy, timings).

### Enabling the artifacts

`MODEL_PATH` is a **startup gate**: in `demo`/`live` mode the file must exist or
`ConfigValidator` aborts with `Model file not found`. Point it at a real artifact:

```env
MODEL_PATH=models/trained/ppo_xauusd.zip
```

`MODE=backtest` skips this check. After training, run `python main.py --check` and then
`start.bat` → `[3] Demo`.

---

## 4. Automation

### Windows Task Scheduler (daily incremental fetch + weekly retrain)

```bat
:: 1) nightly incremental bars, merged into the same CSV (adjust the file name)
schtasks /create /tn "MT5FetchHistory" /sc daily /st 22:30 ^
  /tr "cmd /c cd /d C:\path\to\mt5-ai-xauusd-trader && venv\Scripts\python.exe scripts\fetch_history.py --symbol XAUUSD --timeframe M5 --start 2025-01-01 --out data\raw\XAUUSD_M5.csv >> logs\fetch.log 2>&1"

:: 2) weekly retrain from that CSV
schtasks /create /tn "MT5TrainModels" /sc weekly /d SUN /st 02:00 ^
  /tr "cmd /c cd /d C:\path\to\mt5-ai-xauusd-trader && venv\Scripts\python.exe scripts\train_models.py --data data\raw\XAUUSD_M5.csv --model all >> logs\train.log 2>&1"
```

Or just schedule `train.bat` for the full fetch→train path. Keep the machine logged in with
MT5 running; the MT5 Python API talks to a live terminal session.

### Linux / macOS cron

```cron
# 22:30 daily incremental bars
30 22 * * 1-5 cd /opt/mt5-ai-xauusd-trader && venv/bin/python scripts/fetch_history.py \
  --symbol XAUUSD --timeframe M5 --start 2025-01-01 --out data/raw/XAUUSD_M5.csv >> logs/fetch.log 2>&1
# 02:00 every Sunday retrain
0 2 * * 0 cd /opt/mt5-ai-xauusd-trader && venv/bin/python scripts/train_models.py \
  --data data/raw/XAUUSD_M5.csv --model all >> logs/train.log 2>&1
```

Without a Windows terminal, either use MetaAPI credentials or fetch on Windows and train
anywhere — a CSV is portable.

### GitHub Actions (nightly retrain, no MT5)

```yaml
name: nightly-train
on:
  schedule: [{cron: "0 3 * * *"}]
  workflow_dispatch:
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v5
        with: {python-version: "3.11", cache: pip}
      - run: pip install -r requirements.txt
      - run: python scripts/fetch_history.py --synthetic --bars 20000 --no-merge
      - run: python scripts/train_models.py --data latest --model all --ppo-steps 20000 --lstm-epochs 2 --max-samples 3000
      # upload models/trained or a data CSV artifact for the real run
```

Use synthetic data only as a pipeline health check in CI; production training should use your
broker's real history.

### Retraining cadence

- **Daily**: append bars (`--start` = yesterday) — cheap, keeps the CSV warm.
- **Weekly/monthly**: retrain and compare the report's `val_accuracy` before promoting.
- **Always**: run `python main.py --check` after swapping model files; the model-health gates
  (`MODEL_ACCURACY_FLOOR`, `MODEL_DRIFT_THRESHOLD`) will block trading if quality drops.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `No bars returned` | MT5 not running / wrong symbol / no local history | Log in, open an XAUUSD chart, scroll back, retry; verify `MT5_SERVER` |
| `Configuration Error` on fetch | `.env` missing or placeholders | Run `start.bat` → `[2] 配置向导` (`--setup`) |
| `TA-Lib not installed. Technical indicators will be empty` | C library missing | Windows: install a TA-Lib wheel (`pip install TA-Lib`); Linux: `sudo apt-get install libta-lib0`. Features drop from ~140 to ~13 without it |
| `PyTorch is not installed` | venv incomplete | `venv\Scripts\python.exe -m pip install -r requirements.txt` |
| `PPO training needs torch + stable-baselines3` | same | as above |
| `Model file not found at ...` on startup | `MODEL_PATH` gate | point `MODEL_PATH` at a real artifact or use `MODE=backtest` |
| Training very slow | PPO steps / LSTM epochs / feature count | reduce `--ppo-steps`, `--lstm-epochs`, add `--max-samples`, or `--no-mtf` |

---

**Disclaimer:** trained weights are not shipped with the repository (`models/` is git-ignored).
A model that scores well in-sample can still lose money live — validate with
`python main.py --mode backtest` before enabling `MODE=live`.
