@echo off
cls
:start
cd /d %cd%/src/
python wavelet_analysis.py
set choice=
set /p choice="Do you want to restart? Press 'y' and enter for Yes: "
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='y' goto start