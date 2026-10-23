@echo off
chcp 65001 >nul
REM ============================================================
REM Launch Jupyter with the project virtualenv, rooted at the
REM project directory. This makes both the working directory and
REM the Python kernel correct, so `from src import config` works.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\jupyter-notebook.exe" (
    echo [ERROR] .venv not found or jupyter not installed in it.
    echo Run this first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Jupyter Notebook on port 8889 (project root as home)...
echo.
".venv\Scripts\jupyter-notebook.exe" --port 8889 --no-browser
pause
