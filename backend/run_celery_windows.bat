@echo off
echo Starting Celery for Windows...
cd /d "%~dp0\.."
call .venv\Scripts\activate
python -m celery -A backend.worker.celery_app worker --loglevel=info --pool=solo
pause
