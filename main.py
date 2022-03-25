"""
@authors: Matthew Tarchick, Colleen Kimyewon, Henry Astley

PyWavelets package is citable as:
    
Lee G, Wasilewski F, Gommers R, Wohlfahrt K, O’Leary A, Nahrstaedt H, and Contributors, 
    “PyWavelets - Wavelet Transforms in Python”, 
    2006-, https://github.com/PyWavelets/pywt [Online; accessed 2018-MM-DD].
    
in order to install all files, go to ipython shell and type in
pip instll pycwt

"""


#Extend the folder to include the src
from multiprocessing import dummy
import os, sys #These are used for extending the imports
src_path = os.path.join(os.getcwd(), "src") #This is the folder containing the source codes
sys.path.append(src_path) #add the source path to the system path

#import numpy as np #Import numpy to do matrix processing and linearization
#import cv2 #This is used to interact with all of the image and movies
import matplotlib.pyplot as plt #of course matplotlib and PyPlot are used to do plotting
plt.ioff() #We want to turn the interactive plotting off

from tkinter import * #This is a GUI that will help us interact with the images
from tkinter.filedialog import askopenfilename #This is the file opener which we can use to open the file
root = Tk() #This right away opens the GUI
root.withdraw() #This removes the GUI but still allows the interface to continue. 

#=================Using the PyCWT package to do most of the heavy lifting=================#
import pycwt as cwt
from pycwt.helpers import find


#================================Parsing default arguments================================#
import argparse #This is used to parse extra arguments
parser = argparse.ArgumentParser(description = "Wavlet analysis file outputter")

parser.add_argument('-v', '--verbosity', action = 'count', 
               help = "Levels: 0) nothing 1) function success 2) function progress and details 3) Elapsed time"
               )
parser.add_argument('-lp', '--lowlimit', type = float, help = 'The lowest scale the CWT will use')
parser.add_argument('-hp', '--highlimit', type = float, help = 'The highest scale the CWT will use')
parser.add_argument('-sx', '--smallestscale', help = 'This is the smallest scale in which we will be measuring')
parser.add_argument('-dj', '--suboctaves', help = 'This is the number of suboctaves that we will measure')
parser.add_argument('-j', '--octaves', help = 'This is the number of octaves we will measure')
parser.add_argument('-su', '--samples', help = 'This is the number of samples per unit')
parser.add_argument('-u', '--unit', help = 'This is the unit')
parser.add_argument('-w', '--wavelet', help = 'This is the mother wavelet the analysis will use (Paul, DOG, Morlet, MexicanHat')
parser.add_argument('-f', '--factor', help = 'This is the unit')
parser.add_argument('-g', '--graphing', help = 'Should the wavelet be graphed')

args = parser.parse_args()

per_min = 0.5 #This is the setting for the minimum period 
per_max = 5.0
if args.lowlimit != None:
     per_min = float(args.lowlimit)
if args.highlimit != None:
     per_max = float(args.highlimit)

verbose = 3 #This specifies the level of debug:
#Verbose 1: This tells the minimum 
#Verbose 2: This tells more 
#Verbose 3: This gives all the information
if args.verbosity != None:
     verbose = int(args.verbosity)
     
suboctaves = 1/12 #This is the suboctaves
if args.suboctaves != None:
     suboctaves = int(args.suboctaves)
     
octaves = 10 # This is the number of octaves
if args.octaves != None:
     octaves = int(args.octaves)


J = int(octaves)/suboctaves #This is the tot

wavelet = 'DOG' #This is the default wavelet type, Delay of Gaussian
if args.wavelet != None:
     wavelet = args.wavelet 

order = 2 #This is the default order of the wave
if args.factor != None:
     order = int(args.factor)


su = 78 #two feet
if args.samples != None:
     su = float(args.samples)
     
unit = 'cm' #The default unit that the wavelets will be measured in
if args.unit != None:
     unit = args.unit

graphing = True #Finally we can indicate whether or not we want this to be plotted
if args.graphing != None:
     graphing = args.graphing          

#==================================Imported from SRC==================================#
import selectinwindow as sw #This is a script that is used for slecting in the gui
import boundry_gui as bg #this is used for 
from utilities import * #This imports all of my predefined filtering functions
#print("Imports are completed") #This is a debug statement


if __name__ == '__main__':
     #step 1 will be to select the file name we want to extract from
     if verbose >= 1: print("Select a input file as a video file (.mpg .mp4. avi)")
     input_file = askopenfilename() #'C:/pyscripts/wavelet_analysis/Videos/2018_07_05/GH010222.mp4' 
     print(input_file)
     dirs = input_file.split('/')
     print(dirs)

     #= Include everything under here in a new function file details
     #date = "test" #dirs[-3] #Work in some kind of name split here
     #trial_type = "test" #dirs[-2]
     #name = "test" #dirs[-1]
     #name = "test" #name.split('.')[0]
     #further_split_name = "test" #name.split("_")

     #root.update() #Why is this here?
     #root.destroy()
     a_start = time.time() #Once we are ready to run the analysis we can
     # now we can start to parse the frames

     #in order to do some testing we may need dummy files
     #dummy_image = np.random.rand(100,100)
     #print(dummy_image)
     #collapsed_arr = median_pixel(dummy_image)
     #print(collapsed_arr)
     #plt.plot(collapsed_arr)

