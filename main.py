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
import time
src_path = os.path.join(os.getcwd(), "src") #This is the folder containing the source codes
sys.path.append(src_path) #add the source path to the system path
gui_path = os.path.join(src_path, "gui")
sys.path.append(gui_path) #add the source path to the system path

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
import gui.selectinwindow as sw #This is a script that is used for slecting in the gui
import gui.boundry_gui as bg #this is used for 
from utilities import * #This imports all of my predefined filtering functions
#print("Imports are completed") #This is a debug statement


if __name__ == '__main__':
     #step 1 will be to select the file name we want to extract from
     if verbose >= 1: print("Select a input file as a video file (.mpg .mp4. avi)")
     input_file = askopenfilename() #'C:/pyscripts/wavelet_analysis/Videos/2018_07_05/GH010222.mp4' 
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
     video_capture = cv2.VideoCapture(input_file) #Load the video into the openCV interface
     FRAME_COUNT = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #This is information about the number of frames
     FPS = video_capture.get(cv2.CAP_PROP_FPS) #This is the number of frames per second
     FRAME_HEIGHT  = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #This is the frame height
     FRAME_WIDTH  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) #This is the frame width
     if verbose > 1:
          print("INFO: \n Frame count: ", FRAME_COUNT, "\n",
             "FPS: ", FPS, " \n", 
             "FRAME_HEIGHT: ", FRAME_HEIGHT, " \n",
             "FRAME_WIDTH: ", FRAME_WIDTH, " \n",)

     #set_bbox = False #If the bounding box is not set, we need to establish it
     #=============================Set the bounding box=============================#
     opened_frame, test_frame = video_capture.read()
     bbox = bg.manual_format(test_frame)
     x,y,w,h,angle = bbox
     horizon_begin = x
     horizon_end = x+w
     vert_begin = y
     vert_end = y+h
     #=======================Set the threshold stop light box=======================#
     red_bbox = bg.manual_format(test_frame, stop_sign=True, thet = angle)
     red_x, red_y, red_w, red_h = red_bbox
     box_h_begin = red_x
     box_h_end = red_x+red_w
     box_v_begin = red_y
     box_v_end = red_y + red_h

     indicator = None #This is the red light indicator which sets the recording time
     n_recorded_frames = 0 #This counts the recorded frames
     sig = 0.95 #set the limit for significance

     video_capture.set(2, 0) #reset the position of the video capture to 0
     dropped  = 0 #This keeps track of all the frames that have thrown errors
     for i in range(FRAME_COUNT): #extract the image frame by frame
          success, frame = video_capture.read()
          if success: #If the frame is successfully opened we can continue
               rows, cols, chs = frame.shape #extract the shape of the frame
               real_time = i/FPS #extract the real time

               M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) #we can use the bounding box calculations above
               rot_frame = cv2.warpAffine(frame, M, (cols, rows)) #rotate the frame 
               roi = rot_frame[vert_begin:vert_end,horizon_begin:horizon_end,:] #extract the region of intrest
               red_box = rot_frame[box_v_begin:box_v_end, box_h_begin:box_h_end, 2] #This is the indicator region of intrest, 2 indicates the red channel

               if indicator == None: #if the indicator has not been set yet, set it
                    indicator = np.mean(red_box)
               percent_drop = 1 - (np.mean(red_box)/indicator) #This is the percent difference from the threshold. If this is high enough we drop it 
               if percent_drop >= 0.18: #If the stop sign percent difference is different from the threshold      
                    if verbose >= 1: print('Frame is skipped {} / {}'.format(i, FRAME_COUNT)) 
                    continue
               else:
                    if verbose >= 1: print('Processing frame {} / {}'.format(i+1, FRAME_COUNT))
                    begin_code, data_line = extract_frame(roi) #at this point we will begin the extraction of the line. This returns the raw data line
                    nX = len(data_line) #the number of points 
                    dX = su/nX #the standard units by the amount of time
                    x = np.arange(0, nX) * dX
                    x = x - np.mean(x) #make the x axis left to right
                    var, std, dat_norm = detrend(data_line) #detrend the line

                    #start to set up the wavelet information
                    if wavelet == 'DOG': 
                         mother = cwt.DOG(order)
                    elif wavelet == 'Paul':
                         mother = cwt.Paul(order)
                    elif wavelet == 'Morlet':
                         mother = cwt.Morlet(order)
                    elif wavelet == 'MexicanHat':
                         mother = cwt.MexicanHat(order)   

                    s0 = 4 * dX #se the limit for the wavelet analysis
                    try: #set the alpha value. Sometimes this does not work
                         alpha, _, _ = cwt.ar1(dat_norm)
                    except:
                         alpha = 0.95
                    
                    #do the wavelet transform. Extract the wavelet domain, scales, frequencies, fft, and fft freqs
                    wave, scales, freqs, coi, fft, fftfreqs = cwt.cwt(dat_norm, dX, suboctaves, s0, J, mother)
                    #reconstruct the wave from the infomation
                    iwave = cwt.icwt(wave, scales, dX, suboctaves, mother) * std #This is a reconstruction of the wave
                    power = (np.abs(wave)) ** 2 #This is the power spectra
                    fft_power = np.abs(fft) ** 2 #This is the fourier power 
                    period = 1 / freqs #This is the periods of the wavelet analysis in cm
                    power /= scales[:, None] #This is an option suggested by Liu et. al.
                    
                    #Next we calculate the significance of the power spectra. Significane where power / sig95 > 1
                    signif, fft_theor = cwt.significance(1.0, dX, scales, 0, alpha,
                                                            significance_level=0.95,
                                                            wavelet=mother)
                    sig95 = np.ones([1, nX]) * signif[:, None]
                    sig95 = power / sig95
                    
                    #This is the significance of the global wave
                    glbl_power = power.mean(axis=1)
                    dof = nX - scales  # Correction for padding at edges
                    glbl_signif, tmp = cwt.significance(var, dX, scales, 1, alpha,
                                                            significance_level=0.95, dof=dof,
                                                            wavelet=mother)
                    
                    sel = find((period >= per_min) & (period < per_max))
                    Cdelta = mother.cdelta
                    print(scales)
                    scale_avg = (scales * np.ones((N, 1))).transpose()
                    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
                    #scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)

                    #scale_array[i,:] = scale_array[i,:]/np.max(scale_array[i,:])
                    #data_array[i,:] = data_array[i,:]/np.max(data_array[i,:])
                         
                    scale_avg = var * suboctaves * dX / Cdelta * scale_avg[sel, :].sum(axis=0)
                    scale_avg_signif, tmp = cwt.significance(var, dX, scales, 2, alpha,
                                                                 significance_level=0.95,
                                                                 dof=[scales[sel[0]],
                                                                      scales[sel[-1]]],
                                                                 wavelet=mother)
                    n_recorded_frames += 1
                    if verbose >= 1: print("Frame {} at time {} was successful".format(i+1, real_time)) 