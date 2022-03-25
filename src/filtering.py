"""
From each image, you will get a range of maximum pixels. 
We found that the easiest way to calculate the best pixel to use was to take the median value
This function takes in an image and calcuates the largest nonzero values
"""
def median_pixel(n, arr):
    med_arr = np.zeros(n)
    for i in range(n):
        vals = arr[:,i].nonzero()
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


print("Filtering functions successfully extracted")