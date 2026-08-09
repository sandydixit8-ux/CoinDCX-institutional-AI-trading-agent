@echo off
cd /d C:\TradingAgent
start "" "C:\TradingAgent\.venv\Scripts\python.exe" paper_daemon.py --interval 120 --dry 2 --config config_practice3.json --instance 1h75 --data-dir C:\TradingAgent\data\practice_1h75
