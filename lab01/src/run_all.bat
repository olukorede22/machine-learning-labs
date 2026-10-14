@echo off
cd /d %~dp0
if not exist ..\results mkdir ..\results
python print_versions.py > ..\results\versions.txt
python baseline.py
python measure.py
python budget.py
