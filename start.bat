@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title MT5 AI/ML XAUUSD Trader - 一键启动
cd /d "%~dp0"

REM ===========================================================================
REM  MT5 AI/ML XAUUSD Trader - Windows 一键启动脚本
REM  ---------------------------------------------------------------------------
REM  双击 start.bat         -> 交互式菜单（推荐）
REM  start.bat demo         -> Demo 模拟盘交易 (XAUUSD / ensemble)
REM  start.bat live         -> Live 实盘交易（需二次确认）
REM  start.bat backtest     -> 回测（可追加 --start 2024-01-01 --end 2025-12-31）
REM  start.bat install      -> 强制重新安装依赖
REM  start.bat setup        -> 交互式配置向导（写入 .env）
REM  start.bat doctor       -> 环境诊断
REM  start.bat check        -> 启动前健康检查
REM  start.bat synthetic    -> 合成数据策略演示（无需配置 / 无需 MT5）
REM  start.bat rl           -> 合成环境 RL 评估演示
REM  start.bat test         -> 运行单元测试
REM  start.bat help         -> 显示 main.py 帮助
REM  start.bat --mode demo  -> 其余参数原样传给 main.py
REM ===========================================================================

set "APP_TITLE=MT5 AI/ML XAUUSD Trader"
set "ROOT=%~dp0"
set "VENV_DIR=%ROOT%venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "STAMP=%VENV_DIR%\.deps_ok"
set "ENV_FILE=%ROOT%.env"
set "REQ_FILE=%ROOT%requirements.txt"

REM Python 侧统一使用 UTF-8，避免中文日志乱码
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%ROOT%"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

set "SYS_PY="
set "EXTRA="
set "INTERACTIVE=0"

call :banner
call :prepare
if errorlevel 1 goto :abort

if not "%~1"=="" goto :dispatch
set "INTERACTIVE=1"
goto :menu

REM ===========================================================================
REM  准备阶段：Python -> venv -> 目录 -> .env -> 依赖
REM ===========================================================================
:banner
echo.
echo ================================================================
echo    %APP_TITLE%
echo    Windows 一键启动脚本
echo ================================================================
echo    Demo 模拟盘 / Backtest 回测 / Live 实盘
echo    Python 3.11 + MetaTrader 5
echo ================================================================
echo.
exit /b 0

:prepare
echo [检查] 正在查找可用的 Python 3.11 ...
call :find_python
if not defined SYS_PY goto :no_python

call :verify_python
if errorlevel 1 exit /b 1

call :ensure_venv
if errorlevel 1 exit /b 1

call :ensure_dirs
call :ensure_env
if errorlevel 1 exit /b 1
call :ensure_deps
if errorlevel 1 exit /b 1

echo.
exit /b 0

:find_python
set "SYS_PY="
REM 1) 官方 Python Launcher 指定的 3.11
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3.11"
    exit /b 0
)
REM 2) PATH 中的 python / python3（逐个候选检测，跳过应用商店占位程序）
call :scan_python python
if defined SYS_PY exit /b 0
call :scan_python python3
exit /b 0

:scan_python
for /f "delims=" %%P in ('where %1 2^>nul') do (
    if not defined SYS_PY call :accept_python "%%~fP"
)
exit /b 0

:accept_python
set "CAND=%~1"
echo "!CAND!" | findstr /i "WindowsApps" >nul
if not errorlevel 1 exit /b 0
"!CAND!" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
set "SYS_PY="!CAND!""
exit /b 0

:verify_python
%SYS_PY% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [检查] 已找到 Python 3.11: %SYS_PY%
    exit /b 0
)
echo [警告] 当前 Python 不是 3.11 版本:
%SYS_PY% --version
echo [警告] 本项目依赖锁定 Python 3.11 (torch==2.15.0+cpu 等), 其它版本可能安装失败.
echo        推荐下载: https://www.python.org/downloads/release/python-3119/
set "GO_ON="
set /p "GO_ON=仍要继续尝试? [y/N]: "
if /i not "!GO_ON!"=="y" exit /b 1
exit /b 0

:ensure_venv
if not exist "%VENV_PY%" goto :venv_create
"%VENV_PY%" -c "import sys" >nul 2>&1
if errorlevel 1 goto :venv_broken
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [检查] 虚拟环境就绪: venv
    exit /b 0
)
echo [警告] 现有 venv 使用的 Python 版本不是 3.11.
set "RECREATE="
set /p "RECREATE=是否删除并重建 venv? [y/N]: "
if /i not "!RECREATE!"=="y" (
    echo [提示] 继续使用现有 venv（依赖安装可能失败）.
    exit /b 0
)
echo [环境] 正在删除旧的虚拟环境 ...
rmdir /s /q "%VENV_DIR%"

:venv_create
echo [环境] 正在创建虚拟环境 venv (首次运行) ...
%SYS_PY% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_fail
if not exist "%VENV_PY%" goto :venv_fail
echo [环境] 虚拟环境创建完成.
exit /b 0

:venv_broken
echo [错误] venv 已损坏, 无法运行: %VENV_PY%
echo        请删除 venv 目录后重新运行本脚本.
exit /b 1

:venv_fail
echo [错误] 创建虚拟环境失败.
echo        请确认已安装 Python 3.11, 并在安装时勾选 "Add python.exe to PATH".
exit /b 1

:no_python
echo.
echo [错误] 未检测到可用的 Python.
echo        本系统需要 Python 3.11 (64 位), 请先安装:
echo        https://www.python.org/downloads/release/python-3119/
echo        安装时务必勾选 "Add python.exe to PATH".
set "OPEN_DL="
set /p "OPEN_DL=是否打开下载页面? [Y/n]: "
if /i "!OPEN_DL!"=="n" exit /b 1
start "" "https://www.python.org/downloads/release/python-3119/"
exit /b 1

:ensure_dirs
for %%D in (data logs reports "models\trained") do (
    if not exist "%ROOT%%%~D" mkdir "%ROOT%%%~D" >nul 2>&1
)
echo [检查] 运行目录就绪: data / logs / reports / models\trained
exit /b 0

:ensure_env
if exist "%ENV_FILE%" (
    echo [检查] 已检测到 .env 配置文件.
    exit /b 0
)
if not exist "%ROOT%.env.example" (
    echo [警告] 未找到 .env.example, 跳过 .env 创建.
    exit /b 0
)
echo [配置] 未找到 .env, 正在从 .env.example 创建 ...
copy /y "%ROOT%.env.example" "%ENV_FILE%" >nul
if errorlevel 1 (
    echo [错误] 无法创建 .env 文件, 请检查目录写入权限.
    exit /b 1
)
REM 模板里的 PostgreSQL 占位地址会触发启动校验的 CRITICAL 错误,
REM 这里改写成本地 SQLite（.env 中的重复键以最后一行为准）。
where powershell >nul 2>&1
if errorlevel 1 goto :env_append
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = '%~dp0.env'; $t = Get-Content -LiteralPath $p -Raw; $t = $t -replace '(?m)^DATABASE_URL=.*', 'DATABASE_URL=sqlite:///data/trades.db'; [System.IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false)))" >nul 2>&1
findstr /i /c:"DATABASE_URL=sqlite" "%ENV_FILE%" >nul 2>&1
if not errorlevel 1 goto :env_done

:env_append
>>"%ENV_FILE%" echo DATABASE_URL=sqlite:///data/trades.db

:env_done
echo [配置] .env 已创建, 数据库默认使用 SQLite (data\trades.db).
echo [提示] 请通过菜单 [2] 配置向导或直接编辑 .env 填写 MT5 账号 / 密码 / 服务器.
exit /b 0

:ensure_deps
if not exist "%REQ_FILE%" (
    echo [警告] 未找到 requirements.txt, 跳过依赖安装.
    exit /b 0
)
set "NEED_INSTALL=1"
if not exist "%STAMP%" goto :deps_install
set "STAMP_TIME="
for /f "usebackq delims=" %%T in ("%STAMP%") do set "STAMP_TIME=%%T"
set "REQ_TIME="
for %%F in ("%REQ_FILE%") do set "REQ_TIME=%%~tF"
if /i "!STAMP_TIME!"=="!REQ_TIME!" set "NEED_INSTALL=0"

:deps_install
if "!NEED_INSTALL!"=="0" (
    echo [检查] 依赖已安装, requirements.txt 未变化.
    exit /b 0
)
echo [环境] 首次运行需要安装依赖, 约 5-15 分钟 (含 PyTorch CPU 包), 请保持网络畅通 ...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
echo.
echo [环境] 安装核心诊断库 (rich / pydantic / structlog) ...
"%VENV_PY%" -m pip install rich pydantic pydantic-settings structlog python-dotenv
echo.
echo [环境] 安装 requirements.txt ...
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :deps_fallback
goto :deps_stamp

:deps_fallback
echo.
echo [警告] 完整依赖安装失败, 尝试跳过 TA-Lib 重试 ...
findstr /v /i "TA-Lib" "%REQ_FILE%" > "%TEMP%\mt5_req_notalib.txt"
"%VENV_PY%" -m pip install -r "%TEMP%\mt5_req_notalib.txt"
if errorlevel 1 goto :deps_fail
echo [环境] 单独安装 TA-Lib ...
"%VENV_PY%" -m pip install TA-Lib
if errorlevel 1 echo [警告] TA-Lib 安装失败, 系统将使用内置指标实现.

:deps_stamp
set "REQ_TIME="
for %%F in ("%REQ_FILE%") do set "REQ_TIME=%%~tF"
> "%STAMP%" echo !REQ_TIME!
echo [环境] 依赖安装完成.
exit /b 0

:deps_fail
echo.
echo [错误] 依赖安装失败, 请依次检查:
echo        1) 网络是否可访问 pypi.org 与 download.pytorch.org
echo        2) Python 版本是否为 3.11 (python --version)
echo        3) 手动重试: venv\Scripts\python.exe -m pip install -r requirements.txt
exit /b 1

REM ===========================================================================
REM  交互式菜单
REM ===========================================================================
:menu
echo.
echo ----------------------------------------------------------------
echo    请选择要执行的操作
echo ----------------------------------------------------------------
echo    [1] 环境诊断 (doctor)        首次使用建议先执行
echo    [2] 配置向导 (setup)         填写 MT5 账号 / 密码 / 服务器
echo    [3] Demo 模拟盘交易          XAUUSD / ensemble / demo 模式
echo    [4] 回测 Backtest            历史数据回放
echo    [5] Live 实盘交易            需二次确认, 风险自负
echo    [6] 启动前健康检查 (--check)
echo    [7] 合成数据演示             无需 .env / 无需 MT5
echo    [8] 重新安装依赖             修复依赖安装问题
echo    [9] 查看完整帮助 (--help)
echo   [10] 数据+模型自动更新        缺数据就抓取, 缺模型就训练
echo    [0] 退出
echo ----------------------------------------------------------------
set "EXTRA="
set "CHOICE="
set /p "CHOICE=请输入序号后回车: "
if not defined CHOICE goto :menu
if "%CHOICE%"=="1" goto :m_doctor
if "%CHOICE%"=="2" goto :m_setup
if "%CHOICE%"=="3" goto :m_demo
if "%CHOICE%"=="4" goto :m_backtest
if "%CHOICE%"=="5" goto :m_live
if "%CHOICE%"=="6" goto :m_check
if "%CHOICE%"=="7" goto :m_synthetic
if "%CHOICE%"=="8" goto :m_reinstall
if "%CHOICE%"=="9" goto :m_help
if "%CHOICE%"=="10" goto :m_auto
if "%CHOICE%"=="0" goto :bye
echo [无效] 输入的序号: %CHOICE%
goto :menu

:m_doctor
call :run_doctor
call :pause_back
goto :menu

:m_setup
call :run_setup
call :pause_back
goto :menu

:m_demo
call :run_demo
call :pause_back
goto :menu

:m_backtest
call :run_backtest
call :pause_back
goto :menu

:m_live
call :run_live
call :pause_back
goto :menu

:m_check
call :run_check
call :pause_back
goto :menu

:m_synthetic
call :run_synthetic
call :pause_back
goto :menu

:m_reinstall
echo.
echo [环境] 强制重新安装依赖 ...
if exist "%STAMP%" del /q "%STAMP%" >nul 2>&1
call :ensure_deps
call :pause_back
goto :menu

:m_help
call :run_help
call :pause_back
goto :menu

:m_auto
call :run_auto
call :pause_back
goto :menu

:pause_back
echo.
echo 按任意键返回主菜单...
pause >nul
exit /b 0

REM ===========================================================================
REM  命令行参数分发
REM ===========================================================================
:dispatch
set "CMD=%~1"
set "EXTRA="

:collect_extra
if "%~2"=="" goto :collected
set "EXTRA=!EXTRA! %~2"
shift
goto :collect_extra

:collected
if /i "%CMD%"=="install" goto :x_install
if /i "%CMD%"=="doctor" goto :x_doctor
if /i "%CMD%"=="setup" goto :x_setup
if /i "%CMD%"=="check" goto :x_check
if /i "%CMD%"=="demo" goto :x_demo
if /i "%CMD%"=="live" goto :x_live
if /i "%CMD%"=="backtest" goto :x_backtest
if /i "%CMD%"=="synthetic" goto :x_synthetic
if /i "%CMD%"=="rl" goto :x_rl
if /i "%CMD%"=="test" goto :x_test
if /i "%CMD%"=="help" goto :x_help
if /i "%CMD%"=="auto" goto :x_auto
if /i "%CMD%"=="menu" goto :menu
echo [运行] 参数原样传给 main.py: %*
"%VENV_PY%" "main.py" %*
goto :executed

:x_install
echo [环境] 强制重新安装依赖 ...
if exist "%STAMP%" del /q "%STAMP%" >nul 2>&1
call :ensure_deps
goto :executed

:x_doctor
call :run_doctor
goto :executed

:x_setup
call :run_setup
goto :executed

:x_check
call :run_check
goto :executed

:x_demo
call :run_demo
goto :executed

:x_live
call :run_live
goto :executed

:x_backtest
call :run_backtest
goto :executed

:x_synthetic
call :run_synthetic
goto :executed

:x_rl
call :run_rl
goto :executed

:x_test
echo.
echo [测试] 运行单元测试 (pytest) ...
"%VENV_PY%" -m pytest tests\ -q
goto :executed

:x_help
call :run_help
goto :executed

:x_auto
call :run_auto
goto :executed

REM ===========================================================================
REM  动作实现
REM ===========================================================================
:run_doctor
echo.
echo [诊断] 正在检查环境 / 依赖 / 配置 / 连通性 ...
echo ----------------------------------------------------------------
"%VENV_PY%" "scripts\doctor.py"
set "RC=!errorlevel!"
echo ----------------------------------------------------------------
if not "!RC!"=="0" echo [提示] doctor 报告了需要关注的项目, 请查看上方表格并逐项修复.
exit /b 0

:run_setup
echo.
echo [配置] 正在启动交互式配置向导 ...
"%VENV_PY%" "main.py" --setup
exit /b 0

:run_help
echo.
"%VENV_PY%" "main.py" --help
exit /b 0

:run_auto
echo.
echo [更新] 检查历史数据与 AI 模型 (缺少或过期才动作) ...
echo ----------------------------------------------------------------
"%VENV_PY%" "scripts\auto_pipeline.py" --ensure !EXTRA!
set "RC=!errorlevel!"
echo ----------------------------------------------------------------
if not "!RC!"=="0" echo [提示] 自动更新未完成, 请查看上方日志 (常见原因: 未安装 PyTorch / MT5 未连接).
exit /b 0

:run_check
call :warn_credentials
echo.
echo [检查] 启动前健康检查 (连通性 / 数据库 / 模型) ...
"%VENV_PY%" "main.py" --check !EXTRA!
exit /b 0

:run_demo
call :warn_credentials
echo.
echo [运行] Demo 模拟盘交易: --mode demo --symbol XAUUSD --algo ensemble
"%VENV_PY%" "main.py" --mode demo --symbol XAUUSD --algo ensemble !EXTRA!
exit /b 0

:run_backtest
if defined EXTRA goto :backtest_exec
echo.
echo [提示] 直接回车使用程序默认区间 (2017-01-01 ~ 2026-03-30)
set "BT_START="
set "BT_END="
set /p "BT_START=起始日期 (YYYY-MM-DD): "
set /p "BT_END=结束日期 (YYYY-MM-DD): "
if defined BT_START set "EXTRA=!EXTRA! --start !BT_START!"
if defined BT_END set "EXTRA=!EXTRA! --end !BT_END!"

:backtest_exec
echo.
echo [运行] 回测: --mode backtest --symbol XAUUSD --algo ensemble !EXTRA!
"%VENV_PY%" "main.py" --mode backtest --symbol XAUUSD --algo ensemble !EXTRA!
exit /b 0

:run_live
echo.
echo ###############################################################
echo #  警告: LIVE 实盘模式将使用真实资金交易.
echo #  请先完成回测与 Demo 验证, 实盘盈亏风险由您自行承担.
echo ###############################################################
call :warn_credentials
set "LIVE_OK="
set /p "LIVE_OK=请输入 YES 确认进入实盘 (直接回车取消): "
if /i not "!LIVE_OK!"=="YES" (
    echo [取消] 已取消实盘交易.
    exit /b 0
)
echo.
echo [运行] 实盘交易: --mode live --symbol XAUUSD --algo ensemble --confirm-live
"%VENV_PY%" "main.py" --mode live --symbol XAUUSD --algo ensemble --confirm-live !EXTRA!
exit /b 0

:run_synthetic
echo.
echo [演示] 合成数据策略基准演示 (无需配置) ...
"%VENV_PY%" "src\research\benchmark_demo.py"
exit /b 0

:run_rl
echo.
echo [演示] 合成环境 RL 评估演示 (无需配置) ...
"%VENV_PY%" "src\research\rl_evaluation_demo.py"
exit /b 0

:warn_credentials
if not exist "%ENV_FILE%" exit /b 0
findstr /i /c:"your_password_here" "%ENV_FILE%" >nul 2>&1
if errorlevel 1 exit /b 0
echo.
echo [提示] .env 中的 MT5 账号信息仍为模板占位值 (your_password_here).
echo        建议先运行菜单 [2] 配置向导 (setup), 填写真实 Demo 账号后再启动.
exit /b 0

REM ===========================================================================
REM  收尾
REM ===========================================================================
:executed
set "RC=!errorlevel!"
goto :done

:done
echo.
if "!RC!"=="0" (
    echo [完成] 程序已退出, 返回码: !RC!
) else (
    echo [结束] 程序返回码: !RC!
)
if "!INTERACTIVE!"=="1" (
    echo 按任意键返回主菜单...
    pause >nul
    goto :menu
)
endlocal & exit /b !RC!

:abort
echo.
echo [失败] 初始化未完成, 无法启动.
echo 按任意键退出...
pause >nul
endlocal & exit /b 1

:bye
echo.
echo 再见, 祝交易顺利.
endlocal & exit /b 0
