# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:47:41 2018

@author: Matthew Tarchick
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

lowest = 0
highest = 28
widths = 50

phi = lambda t, f: 2 * np.pi * f * t
wav = lambda t, a, f: a * np.sin(phi(t, f))


if __name__ == "__main__":
    time = np.arange(0, 200)
    vec_arr = wav(time, 110, 0.5) +  signal.gausspulse(time - 0.5, fc = 0.5) * 110
    scales = np.linspace(lowest, highest , widths)
    cfs, frequencies = pywt.cwt(vec_arr, scales, 'morl')
    levels = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8])
    period = 1./frequencies
    power = abs(cfs) ** 2 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
    ax1.plot(vec_arr)
    ax2.contourf(time, np.log2(period), np.log2(power), np.log2(levels))
    plt.show()