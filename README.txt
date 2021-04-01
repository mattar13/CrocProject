*Note this works for windows. 
*********************README********************
*1) Download file wavelet_analysis.zip
*
*2) Extract to destination of your choosing 
*
*3) Download and install Anaconda
*	a) https://www.anaconda.com/download/
*
*4) Ensure ffmpeg is installed
*	a) visit https://www.ffmpeg.org/download.html
*	b) pick the correct build
*	c) into the cmd paste this code:
*		i) setx /M PATH "path\to\ffmpeg\bin;%PATH%"
*		i) replace path\to\ with the preferred path of ffmpeg
*	
*4) Make sure all dependencies are installed:
*	a) numpy
*	b) openCV2
*	c) scipy
	d) pycwt
*
*5) Edit (or make) the default_blank batch file to include these directories, save it as default_settings:
*	@echo off
*	cls
*	:start
*	cd /d c:/THE DIRECTORY OF THE ANALYSIS SCRIPT
*	python wavelet_analysis.py
*	set choice=
*	set /p choice="Do you want to restart? Press 'y' and enter for Yes: "
*	if not '%choice%'=='' set choice=%choice:~0,1%
*	if '%choice%'=='y' goto start
*
*6) Double click on the batch file you have just saved and it will prompt you to enter the file
*	
*
*7) If you want to use a different options, create a new batch-file and specify the options 
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
 