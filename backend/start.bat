@echo off
echo Starting DeepScan Backend...
cd /d "%~dp0"
call .venv\Scripts\activate
start "" "C:\Users\vipin\Downloads\deepfake-detector\frontend\index.html"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause