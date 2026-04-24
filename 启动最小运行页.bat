@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe"
set "APP=%~dp0webapp\minimal_app.py"

if exist "%PYTHON%" (
    "%PYTHON%" -m streamlit run "%APP%"
) else (
    python -m streamlit run "%APP%"
)

echo.
echo Streamlit has stopped.
pause
