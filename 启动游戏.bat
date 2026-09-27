@echo off
cd /d "%~dp0"
set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" main.py
if errorlevel 1 pause
