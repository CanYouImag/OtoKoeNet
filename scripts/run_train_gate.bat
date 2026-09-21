@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe -u -m otokoenet.train --config configs/basic5000_gate.yaml >> log/a1_gate.log 2>&1