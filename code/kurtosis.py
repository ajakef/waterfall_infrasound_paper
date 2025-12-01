#%% imports
import numpy as np
import pandas as pd
import obspy
import gemlog
import os
import matplotlib.pyplot as plt
import riversound
from waterfall_functions import *
from scipy.stats import kurtosis
from scipy.signal import detrend
import obspy.signal.cross_correlation
from obspy.clients.nrl import NRL



#%%
def clip_sg(x):
    low = np.quantile(x, 0.02)
    high = np.quantile(x, 0.98)
    x[x<low] = low
    x[x>high] = high
    return x

def norm(x, r=None):
    x = detrend(x, type='linear')
    #return x/x.std()
    #x = 0.5 * x/np.max(np.abs(x))
    if r is None:
        x = x/np.sqrt(np.median(x**2))
    else:
        x = x/r
    return x

def dB(x): return 10 * np.log10(x)
#%%
st = obspy.read('../data/kurtosis_mseeds/2024-05-20T14*2..HDF.mseed')
st.sort() # 062, the noisy one, comes first
t1 = obspy.UTCDateTime('2024-05-20T19:00')
t2 = obspy.UTCDateTime('2024-05-20T19:05')#05
st = st.detrend('linear')
st.filter('highpass', freq = 1)
st.trim(t1,t2)
df = pd.read_excel("../data/waterfall_summary.ods", engine="odf", skiprows=1).convert_dtypes()
for tr in st: 
    tr.data = tr.data * 3.5012e-3
#%%
def f(st_tmp): # function to calculate sd, kurtosis, xc, and spectrogram for each window
    eps = 1e-6
    st_tmp.detrend('linear')
    output_dict = {
        'sd':st_tmp[0].std(),
        'k0':kurtosis(st_tmp[0]),
        'k1':kurtosis(st_tmp[1])
    }
    ## need to apply window function after calculating stats, before xc and sg
    st_tmp.taper(0.5, 'hamming')
    output_dict['xc'] = obspy.signal.cross_correlation.correlate(st_tmp[0], st_tmp[1], 20)
    #output_dict['sg'] = np.abs(np.fft.fft(st_tmp[0]))**2
    output_dict['sg'] = riversound.pgram(st_tmp[0], 0.01)['spectrum']
    #output_dict['sg1'] = riversound.pgram(st_tmp[1], 0.01)['spectrum']
    return output_dict

win_length_sec = 10
overlap = 0.9
output_dict = riversound.apply_function_windows(st, f, win_length_sec, overlap)

t_mid = output_dict['t_mid'] - t1
st_plot = st.slice(output_dict['t_mid'].min(), output_dict['t_mid'].max())
xc = output_dict['xc']
k0 = output_dict['k0']
k1 = output_dict['k1']
sg = output_dict['sg']
#sg = sg[:,:sg.shape[1]//2]
t_tr = np.arange(len(st_plot[0].data))*0.01
xmin = t_tr.min()
xmax = t_tr.max()
freq = 1/win_length_sec * np.arange(sg.shape[1])
#%%
#####################################################
plt.close('all')
plt.figure(figsize = (6.5, 5)) # width by height, inches

plt.subplot(4,2,1)
plt.plot(t_tr, norm(st_plot[1].data, st_plot[0].data.max()))
plt.plot(t_tr, norm(st_plot[0].data, st_plot[0].data.max()) + 1)

plt.xlim(xmin, xmax)
plt.ylabel('sensor #')
plt.yticks([1,0], [1,2])
plt.xticks([])
plt.title('a. Infrasound Time Series', loc = 'left')

plt.subplot(4,2,3)
w = freq < 40
riversound.image(np.log(clip_sg(sg[:,w])), t_mid, freq[w], log_y = True)
plt.xlim(xmin, xmax)
plt.ylim(np.log10([0.5, 30]))
plt.xticks([])
plt.ylabel('Frequency (Hz)')
plt.title('b. Spectrogram (sensor 1)', loc = 'left')

plt.subplot(4,2,5)
#plt.plot(xc.max(1))
plt.semilogy(t_mid, k1+3)
plt.semilogy(t_mid, k0+3)
plt.axhline(2.5, 0,1, color = 'black', linestyle = '--')
plt.axhline(3.5, 0,1, color = 'black', linestyle = '--')
plt.xlim(xmin, xmax)
plt.xticks([])
plt.yticks([3,5,10,20],[3,5,10,20])
plt.ylabel('Kurtosis')
plt.title('c. Window Kurtosis', loc = 'left')
      
plt.subplot(4,2,7)
t_lag = 0.01 * (np.arange(xc.shape[1]) - (xc.shape[1]-1)/2)
riversound.image(xc, t_mid, t_lag)
plt.xlim(xmin, xmax)
plt.ylabel('Lag (s)')
plt.title('d. Correlogram', loc = 'left')
plt.xlabel('Time (s)')

plt.tight_layout()

plt.subplot(1,2,2)
linewidth_list = [3, 2, 2, 1]
color_list = ['black', 'dimgray', 'darkred', 'orange']
linestyle_list = ['solid', 'solid', 'solid', 'solid']
threshold_list = [np.inf, 2.5, 0.5, 0.1]
for threshold, color, linestyle, linewidth in zip(threshold_list, color_list, linestyle_list, linewidth_list):
    sg_sub = sg[np.abs(k0)<threshold,:]
    plt.loglog(freq, sg_sub.mean(0), color = color, linewidth = linewidth, linestyle = linestyle)
    print(sg_sub.shape[0])
#plt.loglog(freq, output_dict['sg1'].mean(0)/6, color = 'blue', linewidth = linewidth, linestyle = linestyle)

plt.xlim(0.5, 30)
plt.ylim(-1.5 + dB(sg_sub.mean(0)[(freq > 5) & (freq < 40)].min()), 1.5 + dB(sg.mean(0).max()))
plt.legend(['all, n = %s' % sg.shape[0]] + ['|k - 3| < %0.1f, n = %i' % (threshold, np.sum(k0 < threshold)) for threshold in threshold_list[1:]])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (Pa$^2$/Hz)', labelpad = -4) # labelpad to make it neater
plt.title('e. Welch spectra (sensor 1)', loc = 'left')
plt.tight_layout()

plt.savefig('../figures/figure_kurtosis_xc_palouse.png')

