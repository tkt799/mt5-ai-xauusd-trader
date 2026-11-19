@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title MT5 AI/ML XAUUSD Trader - 拉取历史数据 + 训练模型
cd /d "%~dp0"

REM ===========================================================================
REM  一键训练脚本 (Windows)
REM  ---------------------------------------------------------------------------
REM  前置: 先运行一次 start.bat 完成 venv 与依赖安装
REM
REM  用法:
REM    train.bat              完整流程: 拉取历史 -> 训练 PPO + LSTM
REM    train.bat synthetic    合成数据冒烟测试 (无需 MT5 / 无 .env)
REM    train.bat quick        只校验数据与特征管线, 不训练
REM    train.bat help         显示帮助
REM ===========================================================================

REM ── 可修改的默认参数 ────────────────────────────────────────────────────────
set "SYMBOL=XAUUSD"
set "TF=M5"
set "START=2023-01-01"
set "END="
set "PPO_STEPS=200000"
set "LSTM_EPOCHS=10"
REM ───────────────────────────────────────────────────────────────────────────

set "PY=%~dp0venv\Scripts\python.exe"

call :banner

if /i "%~1"=="help" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="synthetic" goto :synthetic
if /i "%~1"=="quick" goto :quick
if not "%~1"=="" goto :usage

if not exist "%PY%" goto :no_venv

REM ── 1/2 拉取历史数据 ───────────────────────────────────────────────────────
echo.
echo [1/2] 从 MT5 拉取历史数据: %SYMBOL% %TF% 从 %START% 开始
echo       (需要 MT5 终端已登录, 且 .env 中 MT5 账号配置正确)
set "END_ARG="
if defined END set "END_ARG=--end %END%"
"%PY%" "scripts\fetch_history.py" --symbol %SYMBOL% --timeframe %TF% --start %START% !END_ARG!
if errorlevel 1 goto :fetch_failed

REM ── 2/2 训练模型 ───────────────────────────────────────────────────────────
echo.
echo [2/2] 训练模型: PPO %PPO_STEPS% 步 / LSTM %LSTM_EPOCHS% 轮
"%PY%" "scripts\train_models.py" --data latest --symbol %SYMBOL% --timeframe %TF% ^
    --model all --ppo-steps %PPO_STEPS% --lstm-epochs %LSTM_EPOCHS%
if errorlevel 1 goto :train_failed
goto :success

:synthetic
if not exist "%PY%" goto :no_venv
echo.
echo [1/2] 生成合成数据 (无需 MT5, 无需 .env)
"%PY%" "scripts\fetch_history.py" --synthetic --bars 20000 --symbol %SYMBOL% --timeframe %TF% ^
    --out "data\raw\synthetic_%SYMBOL%_%TF%.csv" --no-merge
if errorlevel 1 goto :fetch_failed
echo.
echo [2/2] 小规模训练, 验证整条管线可用
"%PY%" "scripts\train_models.py" --data "data\raw\synthetic_%SYMBOL%_%TF%.csv" --symbol %SYMBOL% ^
    --timeframe %TF% --model all --ppo-steps 20000 --lstm-epochs 2 --max-samples 3000
if errorlevel 1 goto :train_failed
goto :success

:quick
if not exist "%PY%" goto :no_venv
echo.
echo [检查] 数据 + 特征管线校验 (不训练)
"%PY%" "scripts\fetch_history.py" --synthetic --bars 6000 --symbol %SYMBOL% --timeframe %TF% ^
    --out "data\raw\synthetic_%SYMBOL%_%TF%.csv" --no-merge
if errorlevel 1 goto :fetch_failed
"%PY%" "scripts\train_models.py" --data "data\raw\synthetic_%SYMBOL%_%TF%.csv" --symbol %SYMBOL% ^
    --timeframe %TF% --model all --dry-run
if errorlevel 1 goto :train_failed
echo.
echo [完成] 管线校验通过. 数据来源没问题时用 train.bat 跑真实训练.
echo 按任意键退出...
pause >nul
exit /b 0

:success
echo.
echo ================================================================
echo   训练完成, 产物位于 models\trained\
echo ----------------------------------------------------------------
echo   在 .env 中启用 (MODEL_PATH 必须指向真实存在的文件):
echo       MODEL_PATH=models/trained/ppo_%SYMBOL%
echo       MODEL_PATH=models/trained/lstm_%SYMBOL%
echo   或改用命令行: python main.py --mode demo --model-dir models/trained
echo ================================================================
echo.
echo 提示: Model 文件就绪后, 用 start.bat -> [3] Demo 验证实时链路.
echo 按任意键退出...
pause >nul
exit /b 0

:fetch_failed
echo.
echo [失败] 历史数据拉取失败. 常见原因:
echo   1) MT5 终端未启动或未登录
echo   2) .env 中 MT5_LOGIN / MT5_PASSWORD / MT5_SERVER 不正确
echo   3) 品种名与券商不一致 (有些券商是 XAUUSD.m / GOLD)
echo   4) 终端里没有该周期历史: 打开 XAUUSD 图表向下滚动加载
echo 可先用 train.bat synthetic 验证脚本本身是否正常.
goto :fail

:train_failed
echo.
echo [失败] 模型训练失败, 请查看上方错误信息.
echo 常见原因: 依赖未装全 (torch / stable-baselines3), 或数据量不足.
goto :fail

:no_venv
echo.
echo [错误] 未找到虚拟环境: %PY%
echo        请先双击运行 start.bat 完成首次安装 (或 start.bat install).

:fail
echo.
echo 按任意键退出...
pause >nul
endlocal & exit /b 1

:usage
echo.
echo 用法: train.bat [synthetic^|quick^|help]
echo.
echo   (无参数)   拉取 MT5 历史数据并训练 PPO + LSTM
echo   synthetic  合成数据跑通整条管线 (无 MT5/无 .env)
echo   quick      只校验数据与特征管线
echo.
echo 修改 train.bat 顶部的 SYMBOL / TF / START / PPO_STEPS / LSTM_EPOCHS 可调整参数.
echo 按任意键退出...
pause >nul
endlocal & exit /b 0

:banner
echo.
echo ================================================================
echo    MT5 AI/ML XAUUSD Trader - 一键训练
echo ================================================================
exit /b 0
