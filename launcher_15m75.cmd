@echo off
cd /d C:\TradingAgent
start "" "C:\TradingAgent\.venv\Scripts\python.exe" paper_daemon.py --interval 120 --dry 2 --config config_practice_15m75.json --instance 15m75 --data-dir C:\TradingAgent\data\practice_15m75
