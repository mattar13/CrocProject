@echo off
cls
call %UserProfile%/Anaconda3/Scripts/activate.bat 
:start
cd /d %cd%/src/
%UserProfile%/Anaconda3/python wavelet_analysis.py
set choice=
set /p choice="Do you want to restart? Press 'y' and enter for Yes: "
if not '%choice%'=='' set choice=%choice:~0,1%
if '%choice%'=='y' goto start