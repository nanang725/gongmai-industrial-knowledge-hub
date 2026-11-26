@echo off
echo [开物] 安装依赖并启动...
pip install fastapi uvicorn pymupdf >nul 2>&1
python make_manuals.py
python app.py
pause
