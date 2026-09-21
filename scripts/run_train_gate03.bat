@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -u -m otokoenet.train --config configs/basic5000_gate03.yaml >> log/a1b_gate03.log 2>&1