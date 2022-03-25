import numpy as np #We have to import this into each file we are using

"""
From each image, you will get a range of maximum pixels. 
We found that the easiest way to calculate the best pixel to use was to take the median value
This function takes in an image and calcuates the largest nonzero values

The dim is in terms of 
"""
def median_pixel(arr, dim = 2):
     n = arr.shape[dim-1]
     if dim == 1:
          med_arr = np.zeros(n)
          for i in range(n):
               vals = arr[i,:].nonzero()
               #print(np.median(vals))
               med_arr[i] = np.median(vals)
          return med_arr 
     elif dim == 2:
          med_arr = np.zeros(n)
          for i in range(n):
               print(i)
               vals = arr[:,i].nonzero()
               print(vals)
               #print(np.median(vals))
               med_arr[i] = np.median(vals)
          return med_arr 

"""
Document this function. I assume it is filtering and detrending, but I need to look at this more

"""
def detrend(wav):
     based = wav# - np.mean(wav)
     N = len(wav)
     t = np.arange(0,N)
     ################################[Polynomial]
     #print(N)
     #dat_fit = wav
     #p = np.polyfit(t, dat_fit, 1)
     #filt = dat_fit - np.polyval(p, t)
     #################################[Splining]
     #seq = splrep(t, based, k = 3, s = 5)
     #filt = splev(based, seq)
     #################[Detrending * Normalizing]
     filt = based    
     std_im = filt.std()
     var = std_im**2
     filt = filt/std_im

     dat_norm = filt - np.mean(filt)
     return var, std_im, dat_norm

"""
This function takes the 

These tables can help if we want to extract certain colors
#  [min    ,max    ]
#H [(10-60),(70-80)]
#S [(50-30),(150+) ]
#V [(0-140),(230+) ]
#Best for trial M00
#H [50 ,80 ]
#S [10 ,170]
#V [10,255]
#Best for orange
#H [50 ,80 ]
#S [50 ,170]
#V [20,255]

"""
def extract_frame(img, hsv_mode = True, green = True):

     begin_code = -1
     row, col, ch = img.shape
     if hsv_mode == True: #This is true if we are in HSV mode
          img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert the red green blue image into HSV mode
          if green:
               GREEN_MIN = np.array([50, 50 , 20],np.uint8) #may need to adjust these for the threshold
               GREEN_MAX = np.array([90, 170, 255],np.uint8)
               green_thresh = cv2.inRange(img_hsv, GREEN_MIN, GREEN_MAX)
               plot_green = median_pixel(col, green_thresh)
               #notrend_green = detrend(plot_green)
          if np.isnan(np.sum(plot_green)):
               begin_code = 1
               #If there is a value called NaN, then send a blank line of zeros
               #return np.zeros(len(plot_green))
               plot_green[np.isnan(plot_green)] = np.mean(plot_green[np.isfinite(plot_green)])
               return begin_code, plot_green
          else:
               RED_MIN = np.array([170,0,0],np.uint8)
               RED_MAX = np.array([255, 255, 255],np.uint8)
               return None #yet
     else: 
          return np.argmax(img[:,:,1], 0)
print("Filtering functions successfully extracted")