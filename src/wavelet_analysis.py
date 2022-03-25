# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:00:41 2018

@author: Matthew Tarchick

PyWavelets package is citable as:
    
Lee G, Wasilewski F, Gommers R, Wohlfahrt K, O’Leary A, Nahrstaedt H, and Contributors, 
    “PyWavelets - Wavelet Transforms in Python”, 
    2006-, https://github.com/PyWavelets/pywt [Online; accessed 2018-MM-DD].
    
in order to install all files, go to ipython shell and type in
pip instll pycwt

"""

import subprocess
import time

import selectinwindow

#Splining and filtering
from scipy.interpolate import splev, splrep
import pandas as pd
from openpyxl import load_workbook




#We need to go through and more clearly document what is doing what. Maybe a walk through of each step



def append_data(filename, dataframe, title, Yticks, 
                names = ["N00.*","M00.*", "M01.*", "O.*", "M04.*"], 
                groups = ["Null(No Movement)","Non-Serrated", "Small_Serrated", "Large_Serrated", "Spaced Serrated"]):
    xls = pd.ExcelFile(filename)
    global_spectras = pd.read_excel(xls, "Global_Spectras")
    global_averages = pd.DataFrame()
    n_values = pd.DataFrame()
    
    writer = pd.ExcelWriter(filename)
    global_spectras[title] = pd.Series(dataframe)
    
    global_averages["Log2[Period]"] = global_spectras["Log2[Period]"]
    global_averages["Period"] = global_spectras["Period"]
    #yvals = np.linspace(Yticks.min(), Yticks.max(), len(global_averages['Period']))
    #global_averages['Actual'] = yvals
    #global_spectras['Actual'] = yvals
    for i in range(len(names)):
        filt = global_spectras.filter(regex = (names[i])).mean(1)
        global_averages[str(groups[i])] = pd.Series(filt.values)
        n_values[str(groups[i])] = len(global_spectras.filter(regex = (names[i])))
    
    global_spectras.to_excel(writer, sheet_name = "Global_Spectras")
    global_averages.to_excel(writer, sheet_name = "Global_Averages")
    n_values.to_excel(writer, sheet_name = "n_values")


def extract_frame(img, hsv_mode = True, green = True):
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
    begin_code = -1
    row, col, ch = img.shape
    if hsv_mode == True:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

#def plot_data(data):


def parse_frames(image_file, sig= 0.95, plotting = False, plot_spectra = True):
    """
    No documentation here yet
    """
    cap = cv2.VideoCapture(image_file) #Load the video into the openCV interface
    if verbose: print("Video successfully loaded")
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    if verbose > 1: 
        FRAME_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        FRAME_WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print("INFO: \n Frame count: ", FRAME_COUNT, "\n",
             "FPS: ", FPS, " \n", 
             "FRAME_HEIGHT: ", FRAME_HEIGHT, " \n",
             "FRAME_WIDTH: ", FRAME_WIDTH, " \n",)
    
    split_path = image_file.split("/")
    file_name = split_path[-1].split(".")[0:-1]
    #print(file_name)
    path_list = split_path[0:-1] + ["\\analysis\\"] + file_name
    directory = os.path.join(*path_list)
    #directory = os.getcwd() + '\\analysis\\{}_{}_{}_{}({})_{}_{}_scaled\\'.format(date, trial_type, name, wavelet, order, per_min, per_max)
    if not os.path.exists(directory):
        os.makedirs(directory)
    made = False    
    frame_idx = 0
    idx = 0
    dropped = 0
    skip = True
    thresh = None
    
    df_wav = pd.DataFrame()
    df_auc = pd.DataFrame()
    df_for = pd.DataFrame()
    df_pow = pd.DataFrame()
    
    for i in range(FRAME_COUNT):
        a, img = cap.read()
        if a:
            frame_idx += 1
            
            if made == False:
                #first we need to manually determine the boundaries and angle
                res = bg.manual_format(img)
                #print(res)
                x,y,w,h,angle = res
                horizon_begin = x
                horizon_end = x+w
                vert_begin = y
                vert_end = y+h
                #scale_array = np.zeros((FRAME_COUNT, abs(horizon_begin - horizon_end)))
                #area_time = np.zeros((FRAME_COUNT)) 
                #df[']
                print("Now Select the Red dot")
                red_res = bg.manual_format(img, stop_sign=True)
                red_x, red_y, red_w, red_h = red_res
                box_h_begin = red_x
                box_h_end = red_x+red_w
                box_v_begin = red_y
                box_v_end = red_y + red_h
                made = True
                #dims = (vert_begin, vert_end, horizon_begin, horizon_end)
            
            
            real_time = i/FPS
            rows, cols, chs = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rot_img = cv2.warpAffine(img, M, (cols, rows))
            roi = rot_img[vert_begin:vert_end,horizon_begin:horizon_end,:]
            
            red_box = img[box_v_begin:box_v_end, box_h_begin:box_h_end, 2]
            if thresh == None:
                thresh = np.mean(red_box)
            #print(np.mean(red_box))
            percent_drop = 1 - (np.mean(red_box)/thresh)
            print(percent_drop)
            if percent_drop >= 0.18:            
                #cv2.imshow("Red Image", red_box)
                #cv2.waitKey(0)
                skip = False
                
                
            if skip:
                if verbose >= 1: print('Frame is skipped {} / {}'.format(frame_idx, FRAME_COUNT)) 
                continue
            
            if verbose >= 1: print('Processing frame {} / {}'.format(frame_idx, FRAME_COUNT))
            
            idx += 1
            begin_code, data_line = extract_frame(roi)

            #We need to detrend the data before sending it away
            N = len(data_line)
            dt = su/N
            t = np.arange(0, N) * dt
            t = t - np.mean(t)
                                   
            var, std, dat_norm = detrend(data_line)
            ###################################################################
            if wavelet == 'DOG':
                mother = cwt.DOG(order)
            elif wavelet == 'Paul':
                mother = cwt.Paul(order)
            elif wavelet == 'Morlet':
                mother = cwt.Morlet(order)
            elif wavelet == 'MexicanHat':
                mother = cwt.MexicanHat(order)                                
                
            s0 = 4 * dt
            try:
                alpha, _, _ = cwt.ar1(dat_norm)
            except:
                alpha = 0.95
            
            wave, scales, freqs, coi, fft, fftfreqs = cwt.cwt(dat_norm, dt, dj, s0, J, mother)
            
            iwave = cwt.icwt(wave, scales, dt, dj, mother) * std #This is a reconstruction of the wave
            
            power = (np.abs(wave)) ** 2 #This is the power spectra
            fft_power = np.abs(fft) ** 2 #This is the fourier power 
            period = 1 / freqs #This is the periods of the wavelet analysis in cm
            power /= scales[:, None] #This is an option suggested by Liu et. al.
            
            #Next we calculate the significance of the power spectra. Significane where power / sig95 > 1
            signif, fft_theor = cwt.significance(1.0, dt, scales, 0, alpha,
                                                     significance_level=0.95,
                                                     wavelet=mother)
            sig95 = np.ones([1, N]) * signif[:, None]
            sig95 = power / sig95
            
            #This is the significance of the global wave
            glbl_power = power.mean(axis=1)
            dof = N - scales  # Correction for padding at edges
            glbl_signif, tmp = cwt.significance(var, dt, scales, 1, alpha,
                                                    significance_level=0.95, dof=dof,
                                                    wavelet=mother)
            
            sel = find((period >= per_min) & (period < per_max))
            Cdelta = mother.cdelta
            scale_avg = (scales * np.ones((N, 1))).transpose()
            scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
            #scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)

            #scale_array[i,:] = scale_array[i,:]/np.max(scale_array[i,:])
            #data_array[i,:] = data_array[i,:]/np.max(data_array[i,:])
                
            scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
            scale_avg_signif, tmp = cwt.significance(var, dt, scales, 2, alpha,
                                                         significance_level=0.95,
                                                         dof=[scales[sel[0]],
                                                              scales[sel[-1]]],
                                                         wavelet=mother)
            Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))

            #Start plotting
            if plotting:
                plt.close('all')
                plt.ioff()
                figprops = dict(figsize=(11, 8), dpi=72)
                fig = plt.figure(**figprops)
                
                wx = plt.axes([0.77, 0.75, 0.2, 0.2])
                imz = 0
                for idxy in range(0,len(period), 10):
                    wx.plot(t, mother.psi(t / period[idxy]) + imz, linewidth = 1.5)
                    imz+=1
                wx.xaxis.set_ticklabels([])
                #wx.set_ylim([-10,10])
                # First sub-plot, the original time series anomaly and inverse wavelet
                # transform.
                ax = plt.axes([0.1, 0.75, 0.65, 0.2])
                ax.plot(t, data_line - np.mean(data_line), 'k', label = "Original Data")
                ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5], label = "Reconstructed wave")
                ax.plot(t, dat_norm, '--k', linewidth=1.5, color=[0.5, 0.5, 0.5],label = "Denoised Wave")
                ax.set_title('a) {:10.2f} from beginning of trial.'.format(real_time))
                ax.set_ylabel(r'{} [{}]'.format("Amplitude", unit))
                ax.legend(loc = 1)
                ax.set_ylim([-200, 200])
                #If the non-serrated section, bounds are 200 - 
                # Second sub-plot, the normalized wavelet power spectrum and significance
                # level contour lines and cone of influece hatched area. Note that period
                # scale is logarithmic.
                bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
                levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
                cont = bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                            extend='both', cmap=plt.cm.viridis)
                extent = [t.min(), t.max(), 0, max(period)]
                bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
                        extent=extent)
                bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                                        t[:1] - dt, t[:1] - dt]),
                        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                        np.log2(period[-1:]), [1e-9]]),
                        'k', alpha=0.3, hatch='x')
                bx.set_title('b) {} Octaves Wavelet Power Spectrum [{}({})]'.format(octaves, mother.name, order))
                bx.set_ylabel('Period (cm)')
                #
                Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                        np.ceil(np.log2(period.max())))
                bx.set_yticks(np.log2(Yticks))
                bx.set_yticklabels(Yticks)
                cbar = fig.colorbar(cont, ax = bx)
                # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
                # noise spectra. Note that period scale is logarithmic.
                cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
                cx.plot(glbl_signif, np.log2(period), 'k--')
                cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
                cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
                        linewidth=1.)
                cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
                cx.set_title('c) Global Wavelet Spectrum')
                cx.set_xlabel(r'Power [({})^2]'.format(unit))
                #cx.set_xlim([0, (var*fft_theor).max()])
                plt.xscale('log')
                cx.set_ylim(np.log2([period.min(), period.max()]))
                cx.set_yticks(np.log2(Yticks))
                cx.set_yticklabels(Yticks)
                
                            #if sig_array == []:
                yvals = np.linspace(Yticks.min(), Yticks.max(), len(period))

                
                plt.xscale('linear')
                plt.setp(cx.get_yticklabels(), visible=False)
                
                # Fourth sub-plot, the scale averaged wavelet spectrum.
                dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
                dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
                dx.plot(t, scale_avg, 'k-', linewidth=1.5)
                dx.set_title('d) {}-{}cm scale-averaged power'.format(per_min, per_max))
                dx.set_xlabel('Distance from center(cm)')
                dx.set_ylabel(r'Average variance [{}]'.format(unit))
                #dx.set_ylim([0,500])
                ax.set_xlim([t.min(), t.max()])
                #plt.savefig(directory+'{}_analysis_frame-{}.png'.format(name, idx), bbox = 'tight')
            
            if verbose >= 2:
                print('*'*int((i/FRAME_COUNT)*100))
            
            
            df_wav[real_time] = (pd.Series(dat_norm, index = t))
            df_pow[real_time] = (pd.Series(var * glbl_power, index = np.log2(period)))
            df_for[real_time] = (pd.Series(var * fft_power , index = np.log2(1./fftfreqs)))
            df_auc[real_time] = [np.trapz(data_line)]
        
        else:
            print("Frame #{} has dropped".format(i))
            dropped  += 1

                  
    if verbose >= 1: print('All images saved')
    if verbose >= 1: print("{:10.2f} % of the frames have dropped".format((dropped/FRAME_COUNT)*100))
    
    #Plotting and saving the power spectra
    ##[Writing means to a single file]#########################################
    df_wav['Mean'] = df_wav.mean(axis = 1)
    df_pow['Mean'] = df_pow.mean(axis = 1)
    df_for['Mean'] = df_for.mean(axis = 1)
    df_auc['Mean'] = df_auc.mean(axis = 1)
    
    df_wav['Standard Deviation'] = df_wav.std(axis = 1)
    df_pow['Standard Deviation'] = df_pow.std(axis = 1)
    df_for['Standard Deviation'] = df_for.std(axis = 1)
    df_auc['Standard Deviation'] = df_auc.std(axis = 1)
    
    ##[Writing analysis to excel]##############################################
    
    print("Writing files")
    writer = pd.ExcelWriter(directory + "analysis{}.xlsx".format(trial_name))
    df_wav.to_excel(writer, "Raw Waveforms")
    df_auc.to_excel(writer, "Area Under the Curve")
    df_for.to_excel(writer, "Fourier Spectra")
    df_pow.to_excel(writer, "Global Power Spectra")
    writer.save()


    if plot_spectra:
        row, cols = df_pow.shape
        time = np.arange(0, cols)/FPS
        
        plt.close('all')
        plt.ioff()
        plt.contourf(time, df_pow.index.tolist(), df_pow)
        plt.contour(time, df_pow.index.tolist(), df_pow) 
        plt.title("Global Power over Time")
        plt.ylabel("Period[cm]")
        plt.xlabel("Time")
        cax = plt.gca()
        #plt.xscale('log')
        cax.set_ylim(np.log2([period.min(), period.max()]))
        cax.set_yticks(np.log2(Yticks))
        cax.set_yticklabels(Yticks)
        
        
        plt.savefig(directory+'{}_global_power-{}.png'.format(name, idx), bbox = 'tight')
        
        row, cols = df_for.shape
        time = np.arange(0, cols)/FPS
        plt.close('all')
        plt.ioff()
        plt.contourf(time, df_for.index.tolist(), df_for)
        plt.contour(time, df_for.index.tolist(), df_for) 
        plt.title("Fourier Power over Time")
        plt.ylabel("Period[cm]")
        plt.xlabel("Time")
        cax = plt.gca()
        #plt.xscale('log')
        cax.set_ylim(np.log2([period.min(), period.max()]))
        cax.set_yticks(np.log2(Yticks))
        cax.set_yticklabels(Yticks)
        plt.savefig(directory+'{}_fourier_power-{}.png'.format(name, idx), bbox = 'tight')
        
        plt.close('all')
        plt.ioff()
        rows, cols = df_auc.shape    
        time = np.arange(0, cols)/FPS
        plt.plot(time, df_auc.T)
        plt.xlabel("Time")
        plt.ylabel("Area under the curve in cm")
        plt.title("Area under the curve over time")
        plt.savefig(directory+'{}_area_under_curve-{}.png'.format(name, idx), bbox = 'tight')
        

        
        
        
        #filename = 'C:\\pyscripts\\wavelet_analysis\\Overall_Analysis.xlsx'
        #append_data(filename, df_pow['Mean'].values,  str(trial_name), Yticks)
        ##[Plotting mean power and foruier]########################################
        plt.close('all')
        plt.ioff()
        plt.plot(df_pow['Mean'],  df_pow.index.tolist(), label = "Global Power")    
        plt.plot(df_for['Mean'],  df_for.index.tolist(), label = "Fourier Power")
        plt.title("Global Power averaged over Time")
        plt.ylabel("Period[cm]")
        plt.xlabel("Power[cm^2]")
        cax = plt.gca()
        #plt.xscale('log')
        cax.set_ylim(np.log2([period.min(), period.max()]))
        cax.set_yticks(np.log2(Yticks))
        cax.set_yticklabels(Yticks)
        plt.legend()
        plt.savefig(directory+'{}_both_{}.png'.format(name, idx), bbox = 'tight')
        
        plt.close('all')
        plt.ioff()
        plt.plot(df_pow['Mean'],  df_pow.index.tolist(), label = "Global Power")  
        plt.title("Global Power averaged over Time")
        plt.ylabel("Period[cm]")
        plt.xlabel("Power[cm^2]")
        cax = plt.gca()
        #plt.xscale('log')
        cax.set_ylim(np.log2([period.min(), period.max()]))
        cax.set_yticks(np.log2(Yticks))
        cax.set_yticklabels(Yticks)
        plt.legend()
        plt.savefig(directory+'{}_global_power_{}.png'.format(name, idx), bbox = 'tight')
        
        plt.close('all')
        plt.ioff()   
        plt.plot(df_for['Mean'],  df_for.index.tolist(), label = "Fourier Power")
        plt.title("Fourier averaged over Time")
        plt.ylabel("Period[cm]")
        plt.xlabel("Power[cm^2]")
        cax = plt.gca()
        #plt.xscale('log')
        cax.set_ylim(np.log2([period.min(), period.max()]))
        cax.set_yticks(np.log2(Yticks))
        cax.set_yticklabels(Yticks)
        plt.legend()
        plt.savefig(directory+'{}_fourier_{}.png'.format(name, idx), bbox = 'tight')
        
        cap.release()
    
    print(directory)
    return directory
    
def write_video(directory, verbose, rate = 10):
    open_dir = directory + '{}_analysis_frame-%d.png'.format(name)
    save_dir = directory + '{}_analysis_vid.mp4'.format(name)
    #trim_dir = directory + '{}_analysis_trim.mp4'.format(name)
    subprocess.check_call('ffmpeg -i {} -vcodec mpeg4 -q:v 2 -async 44100 -filter:v \"setpts=4.0*PTS\" -y {}'.format(open_dir, save_dir), shell = True)
    #print(x)
    #subprocess.check_call('ffmpeg -i {} -ss 00:00:10 -t 00:00:15 -y {}'.format(save_dir, trim_dir), shell = True)
   


if __name__ == '__main__':
    print("Beginning data analysis script")
    root = Tk()
    root.withdraw()
    btn_down = False #Need this global variable for the mouse callbacks
    
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
    print("Arguments parsed")
    
    per_min = 0.5
    per_max = 5.0
    if args.lowlimit != None:
        per_min = float(args.lowlimit)
    if args.highlimit != None:
        per_max = float(args.highlimit)
    
    verbose = 3
    if args.verbosity != None:
        verbose = int(args.verbosity)
        
    dj = 1/12
    if args.suboctaves != None:
        dj = int(args.suboctaves)
        
    octaves = 10
    J = octaves/dj
    if args.octaves != None:
        octaves = int(args.octaves)
        J = int(args.octaves)/dj

    su = 78 #two feet
    if args.samples != None:
        su = float(args.samples)
        
    unit = 'cm'
    if args.unit != None:
        unit = args.unit

    wavelet = 'DOG'
    if args.wavelet != None:
        wavelet = args.wavelet  

    order = 2 #TODO: Change back to 2
    if args.factor != None:
        order = int(args.factor)

    graphing = True
    if args.graphing != None:
        graphing = args.graphing          

    if verbose >= 1: print("Select a input file as a video file (.mpg .mp4. avi)")
  
    input_file = askopenfilename()#'C:/pyscripts/wavelet_analysis/Videos/2018_07_05/GH010222.mp4' 
    root.update() #This probably is an issue
    root.destroy()
    
    dirs = input_file.split('/')
    print(dirs)
    date = "test" #dirs[-3]
    trial_type = "test" #dirs[-2]
    name = "test" #dirs[-1]
    name = "test" #name.split('.')[0]
    further_split_name = "test" #name.split("_")
    #Sometimes the trial name may not necessarily have the correct info
    if len(further_split_name) > 4:
        trial_name = str(further_split_name[3]+"_"+further_split_name[4])
    else:
        #for now leave as this 
        trial_name = "Test"

        
    a_start = time.time()
    if verbose >= 1: print("File {} successfully loaded".format(input_file)) 
    #CODE FOR TAKING IN DATA
    im_dir = parse_frames(input_file)
    print(im_dir)
    #if verbose >= 1: print('{} frames successfully parsed'.format(len(frame_to_vec)))
    a_end = time.time()
    if verbose >= 3: print("Total analysis time = {}".format(abs(a_start - a_end)))
    
    #TODO: Determine how to output a csv and what needs to go in the csv
    #CODE FOR OUTPUTTING DATA
    w_start = time.time()
    #write_video(im_dir, verbose) When the analysis is ready to be plotted
    if verbose >= 1: print('Movie file written to {}'.format(input_file))
    w_end = time.time()
    if verbose >= 3: print("Total write time = {}".format(abs(w_start - w_end)))

    #plt.imshow(vec_to_spect[10], aspect = 'auto')
    #plt.show()
    
