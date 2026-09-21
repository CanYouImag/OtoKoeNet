@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -u -m otokoenet.train --config configs/basic5000_base.yaml >> log/a1_base.log 2>&1