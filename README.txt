*Note this works for windows. 
*********************RUNNING FROM BATCH******************************************************* ********************
*1) Download github folder (https://github.com/mattar13/CrocProject).
*    -) Click the green code button
*    -) Click on Download ZIP
*    
*2) Extract to destination of your choosing 
*
*3) Download and install Anaconda
*	a) https://www.anaconda.com/download/
*
*5) Make sure all dependencies are installed by:
*    -) Open Anaconda -> go into Environments
*    -) Click green arrow
*    -) click Open Terminal
*    -) type in commands in brackets
*	     a) numpy [pip install numpy]
*	     b) openCV2 [pip install opencv-python]
*	     c) scipy [pip install scipy]
*	     d) pycwt [pip install pycwt] 
*
*6) Double click on the batch file and follow the prompts to extract the files
* After setting this up, you will only need to click on the batch files to run the analysis

*********************RUNNING MANUALLY*********************************************************************************** 
*1) Open Anaconda. Click on environments -> Click on the green arrow -> Click open terminal
*
*2) Point the terminal to the location you saved the github repo 
*    -) In the terminal type on [cd [WHERE YOU EXTRACTED THE GITHUB REPO]/src]
*
*3) type: [python wavelet_analysis.py]
*
* While this takes fewer steps it has to be done every time you want to analyze a file
*
*************************EXTRA*******************************************************************************************
* If you want to use a different options, create a new batch-file and specify the options 
* EX: for doing a Mexican Hat analysis with a factor of 6
*	@echo off
*	cls
*	:start
*	cd /d c:/THE DIRECTORY OF THE ANALYSIS SCRIPT
*	python wavelet_analysis.py -w MexicanHat -f 6
*	set choice=
*	set /p choice="Do you want to restart? Press 'y' and enter for Yes: "
*	if not '%choice%'=='' set choice=%choice:~0,1%
*	if '%choice%'=='y' goto start 
*SAVE THIS BATCH FILE AS MexicanHat_6.bat
*
*
*-)Here are some available options
*	-lp	for scale wave reconstruction the lower limit the analysis will use (default 0.5)
*	-hp	for scale wave reconstruction the higher limit the analysis will use (default 5)
*	-sx	the smallest scale the analysis will use (default 2x the smallest sampling frequency) 
*	-dj	the amount of suboctaves the analysis will use (default 1/12)
*	-j	the amount of octaves the analysis will use (defualt 10)
*	-su	the real world quantity measured (defualt 78cm)
*	-u	the real world units measured (defualt cm)
*	-w	the mother wavelet that the analysis will use (defualt DOG)
*	-f	the factor of mother wavelet the analysis will use (default 2)
 