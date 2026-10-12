@echo off
chcp 65001 > nul
echo ========================================================
echo    Khoi dong TrafficVision AI (Flask Web Server)
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
pause
