# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:50:01 2018

@author: Matthew Tarchick
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft, ifft
import pyaudio
import IPython.display as ipd
import csv
import cv2
from scipy.interpolate import splev, splrep
from PIL import Image
import math
import pywt
import pycwt as cwt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from pycwt.helpers import find
import pandas as pd
import numpy as np


default_params = {
        'per_pixel':30.48,
        'min_per':1.0,
        'max_per':3.0,
        'sx':0.5,
        'octaves':10,
        'suboctaves':12,
        'units':'cm',
        'order':2 
                  }

def graph_wavelet(data_xs, title, lims, font = 11, params = default_params):
    a_lims, b_lims, d_lims = lims
    plt.rcParams.update({'font.size': font})
    return_data = {}
    
    N = len(data_xs)
    dt = (2*params['per_pixel'])/N #This is how much cm each pixel equals
    t = np.arange(0, N) * dt
    t = t - np.mean(t)
    t0 = 0
    per_min = params['min_per']
    per_max = params['max_per']
    units = params['units']
    sx = params['sx']
    octaves = params['octaves']
    dj = 1/params['suboctaves'] #suboctaves
    order = params['order']
    
    var, std, dat_norm = detrend(data_xs)
    mother = cwt.DOG(order) #This is the Mother Wavelet
    s0 = sx * dt #This is the starting scale, which in out case is two pixels or 0.04cm/40um\
    J = octaves/dj #This is powers of two with dj suboctaves
    
    return_data['var'] = var
    return_data['std'] = std
    
    try:
        alpha, _, _ = cwt.ar1(dat_norm) #This calculates the Lag-1 autocorrelation for red noise
    except: 
        alpha = 0.95
            
    wave, scales, freqs, coi, fft, fftfreqs = cwt.cwt(dat_norm, dt, dj, s0, J,
                                                              mother)
    return_data['scales'] = scales
    return_data['freqs'] = freqs
    return_data['fft'] = fft
    iwave = cwt.icwt(wave, scales, dt, dj, mother) * std
        
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    power /= scales[:, None] #This is an option suggested by Liu et. al.
    

    #Next we calculate the significance of the power spectra. Significane where power / sig95 > 1
    signif, fft_theor = cwt.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
    
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = cwt.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)
    
    sel = find((period >= per_min) & (period < per_max))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = cwt.significance(var, dt, scales, 2, alpha,
                                                 significance_level=0.95,
                                                 dof=[scales[sel[0]],
                                                      scales[sel[-1]]],
                                                 wavelet=mother)
    
    
    # Prepare the figure
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 11), dpi=72)
    fig = plt.figure(**figprops)
    
    wx = plt.axes([0.77, 0.75, 0.2, 0.2])
    imz = 0
    for idxy in range(0,len(period), 10):
        wx.plot(t, mother.psi(t / period[idxy]) + imz, linewidth = 1.5)
        imz+=1
        wx.xaxis.set_ticklabels([])
    
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(t, data_xs, 'k', linewidth=1.5)
    ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(t, dat_norm, '--', linewidth=1.5, color=[0.5, 0.5, 0.5])
    if a_lims != None:
        ax.set_ylim([-a_lims, a_lims])
    ax.set_title('a) {}'.format(title))
    ax.set_ylabel(r'Displacement [{}]'.format(units))
    #ax.set_ylim([-20,20])

    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
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
    
    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, np.log2(period), 'k--')
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
            linewidth=1.)
    
    return_data['global_power'] = var * glbl_power
    return_data['fourier_spectra'] = var * fft_power
    return_data['per'] = np.log2(period)
    return_data['amp'] = np.log2(1./fftfreqs)
    
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Power Spectrum')
    cx.set_xlabel(r'Power [({})^2]'.format(units))
    if b_lims != None:
        cx.set_xlim([0,b_lims])
    #cx.set_xlim([0,max(glbl_power.max(), var*fft_power.max())])
    #print(max(glbl_power.max(), var*fft_power.max()))
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)
    return_data['yticks'] = Yticks
    
    plt.setp(cx.get_yticklabels(), visible=False)
    
    # Fourth sub-plot, the scale averaged wavelet spectrum.
    dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
    dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    dx.plot(t, scale_avg, 'k-', linewidth=1.5)
    dx.set_title('d) {}--{} cm scale-averaged power'.format(per_min, per_max))
    dx.set_xlabel('Displacement (cm)')
    dx.set_ylabel(r'Average variance [{}]'.format(units))
    ax.set_xlim([t.min(), t.max()])
    if d_lims != None:
        dx.set_ylim([0,d_lims])
    plt.savefig("C:\pyscripts\wavelet_analysis\Calibrated Images\{}".format(title))
    return fig, return_data


def median_pixel(n, arr):
    med_arr = np.zeros(n)
    for i in range(n):
        vals = arr[:,i].nonzero()
        #print(np.median(vals))
        med_arr[i] = np.median(vals)
    return med_arr 


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
    
#%%


#%%

phi = lambda t, f: 2 * np.pi * f * t
wav = lambda t, a, f: a * np.sin(phi(t, f)) 

if __name__ == "__main__":
    
    
    setup        = True
    if setup:
        t = np.linspace(-9,10, 1000)
        f = np.linspace(0,200,len(t))
        
        sin_perf1x = wav(t, 110, 0.50)
        sin_perf1x_wide = wav(t, 110, 0.25)
        sin_perf2x = wav(t, 220, 0.50)
        sin_perf4x = wav(t, 440, 0.50)
        
        gaus_1x = signal.gausspulse(t - 0.5, fc = 0.5) * 110
        gaus_1x_wide = signal.gausspulse(t - 0.5, fc = 0.25) * 110
        
        gaus_2x = signal.gausspulse(t - 0.5, fc = 0.5) * 220
        gaus_4x = signal.gausspulse(t - 0.5, fc = 0.5) * 440
        
        
        plt.figure()
        plt.title("Widening of Gauss Pulse")
        plt.plot(t, gaus_1x)
        plt.plot(gaus_1x_wide)
        plt.ylabel("Amplitude")
        plt.xlabel("Space")
        plt.show()
        
        plt.figure()
        plt.title("Widening of a Sine wave")
        plt.plot(sin_perf1x)
        plt.plot(sin_perf1x_wide)
        plt.ylabel("Amplitude")
        plt.xlabel("Space")
        plt.show()
        
        plt.figure()
        plt.title("Increasing amplitude of Gauss Pulse")
        plt.plot(gaus_1x)
        plt.plot(gaus_2x)
        plt.plot(gaus_4x)
        plt.ylabel("Amplitude")
        plt.xlabel("Space")
        plt.show()
        
        
        plt.figure()
        plt.title("Increasing Amplitude of Sine Wave")
        plt.plot(sin_perf1x)
        plt.plot(sin_perf2x)
        plt.plot(sin_perf4x)
        plt.ylabel("Amplitude")
        plt.xlabel("Space")
        plt.show()
        
        plt.figure()
        plt.plot(gaus_1x)
        plt.axis("off")
        plt.show()
        
        
        plt.figure()
        plt.plot(gaus_1x_wide)
        plt.axis("off")
        plt.show()
    
    #%%   Vary the frequency of the Gauss Pulse 1X 
    
    title = "Analysis of Gauss Pulse"
    a_lims = 150
    b_lims = 500_000
    d_lims = 3000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(gaus_1x, title, lims)
    #fig1.show()
    
    #%% Vary the frequency of the Gauss Pulse (2x wide)
    
    title = 'Analysis of Gauss Wide pulse'
    a_lims = 150
    b_lims = 500_000
    d_lims = 3000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(gaus_1x_wide, title, lims)
    #fig1.show()    
          
    #%%Vary Amplitude Gauss 1x
    title = 'Analysis of Gauss 1x Amplitude pulse'
    a_lims = 500
    b_lims = 1_500_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(gaus_1x, title, lims)
    #fig1.show()
    
    #%% Vary Gauss Amplitude 2x
    
    title = 'Analysis of Gauss 2x Amplitude pulse'
    a_lims = 500
    b_lims = 1_500_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(gaus_2x, title, lims)
    #fig1.show()
    
    #%% Vary Gauss Amplitude 4x

    title = 'Analysis of Gauss 4x Amplitude pulse'
    a_lims = 500
    b_lims = 1_500_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(gaus_4x, title, lims)
    #fig1.show()
    
    #%% Vary Sine Freq Narrow
    
    title = 'Analysis of Sine'
    a_lims = 400
    b_lims = 2_500_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(sin_perf1x, title, lims)
    #fig1.show()
    
    #%% Vary Sine Freq Wide
    
    title = 'Analysis of Sine Wide'
    a_lims = 400
    b_lims = 2_500_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(sin_perf1x_wide, title, lims)
    #fig1.show()    
    
    #%% Vary Sine Amplitude 1x
    

    title = 'Analysis of Sine 1x Amp' 
    a_lims = 500
    b_lims = 42_000_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(sin_perf1x, title, lims)
    #fig1.show()  
    
    #%% Vary Sine Amplitude 2x
    
    title = 'Analysis of Sine 2x Amp'  
    a_lims = 500
    b_lims = 42_000_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(sin_perf2x, title, lims)
    #fig1.show() 
    
    #%% Vary Sine Amplitude 4x
    
    title = 'Analysis of Sine 4x Amp'   
    a_lims = 500
    b_lims = 42_000_000
    d_lims = 50_000
    lims = (a_lims, b_lims, d_lims)
    fig1,_ = graph_wavelet(sin_perf4x, title, lims)
    #fig1.show() 
    
    #%%        
       
    gaus_pt1 = signal.gausspulse(t - 0.8, fc = 0.5) * 110
    gaus_pt2 = signal.gausspulse(t - 0.1, fc = 0.2) * 110
    gauss = np.maximum(gaus_pt1 + gaus_pt2, 0)
    
    gaus_pt3 = signal.gausspulse(t - 0.1, fc = 0.5) * 110
    gaus_pt4 = signal.gausspulse(t - 0.8, fc = 0.2) * 110
    gauss2 = np.maximum(0, -(gaus_pt1 + gaus_pt2))
    plt.rcParams.update({'font.size': 12})
    
    #%%
    
    title = 'Analysis of Gauss pulse'
    a_lims = 150
    b_lims = None
    d_lims = 3000
    lims = (a_lims, b_lims, d_lims)
    fig1, ret1 = graph_wavelet(gauss, title, lims, font = 15)
    #fig1.show() 
    
    #%%
    
    title = 'Analysis of Gauss pulse 2'
    a_lims = 150
    b_lims = None
    d_lims = 3000
    lims = (a_lims, b_lims, d_lims)
    fig1, ret2 = graph_wavelet(gauss2, title, lims, font = 15)
    #fig1.show() 
    
    #%% All in one graph
    plt.figure()
    plt.plot(gauss)
    plt.plot(gauss2)
    plt.close('all')
    #%%
    lw = 5
    fig = plt.figure(figsize = (50, 30))
    plt.rcParams.update({'font.size': 40})
    
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(234)
    
    ax3 = plt.subplot(232)
    ax4 = plt.subplot(235)
    
    ax5 = plt.subplot(233)
    ax6 = plt.subplot(236)
    
    freqs1 = ret1['fourier_spectra']
    amps1  = ret1['amp']
    pow1   = ret1['global_power']
    per1   = ret1['per']
    Yticks = ret1['yticks']
    
    freqs2 = ret2['fourier_spectra']
    amps2  = ret2['amp']
    pow2   = ret2['global_power']
    per2   = ret2['per']
    
    ax1.set_title("Gauss Pulse")
    ax1.plot(gauss, linewidth = lw)
    ax1.set_ylabel("Amplitude[unit]")
    ax1.grid()
    
    ax2.plot(gauss2, linewidth = lw)
    ax2.set_ylabel("Amplitude[unit]")
    ax2.set_xlabel("Space[unit]")
    ax2.grid()
    
    ax3.set_title("Fourier Analysis")
    ax3.plot(freqs1/10e3,  amps1, 'k', linewidth = lw)
    ax3.plot(pow1/10e3, per1, color='#cccccc', linewidth = lw)
    ax3.set_yticks(np.log2(Yticks))
    ax3.set_ylabel("Period[unit]")
    ax3.set_yticklabels(Yticks)
    ax3.grid()
    
    ax4.plot(freqs2/10e3, amps2, 'k', linewidth = lw)
    ax4.plot(pow2/10e3, per2, color='#cccccc', linewidth = lw)
    ax4.set_yticks(np.log2(Yticks))
    ax4.set_yticklabels(Yticks)
    ax4.set_ylabel("Period[unit]")
    ax4.set_xlabel("Power[unit^2 x 10e3]")
    ax4.grid()
    
    ax5.set_title("Global Power Spectra")
    ax5.plot(freqs1/10e3, amps1, color='#cccccc', linewidth = lw)
    ax5.plot(pow1/10e3,per1, 'k', linewidth = lw)
    ax5.set_ylabel("Period [unit]")
    ax5.set_yticks(np.log2(Yticks))
    ax5.set_yticklabels(Yticks)
    ax5.grid()
    
    ax6.plot(freqs2/10e3, amps2, color='#cccccc', linewidth = lw)
    ax6.plot(pow2/10e3, per2, 'k', linewidth = lw)
    ax6.set_ylabel("Period [unit]")
    ax6.set_yticks(np.log2(Yticks))
    ax6.set_yticklabels(Yticks)
    ax6.set_xlabel("Power[unit^2 x 10e3]")
    ax6.grid()        
    
    plt.savefig("C:\pyscripts\wavelet_analysis\Calibrated Images\Demo")
    plt.tight_layout()
    plt.show()
    

    
        