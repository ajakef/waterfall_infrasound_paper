import matplotlib.pyplot as plt; import numpy as np; import gemlog; import glob; import pandas as pd; import obspy; import os;
import riversound; import scipy
import sys
os.chdir('/home/jake/Dropbox/StreamAcoustics/waterfall_paper/code')
#sys.path.append('/home/jake/Dropbox/StreamAcoustics/waterfall_paper/code'); 
from waterfall_functions import *

#%% Read the spreadsheet
df = pd.read_excel("../other_data/waterfall_summary.ods", engine="odf", skiprows=1).convert_dtypes()

for col in df.select_dtypes(include="Float64").columns:
    df[col] = df[col].astype(float)
df = df.dropna(how="all")
for i in range(df.shape[0]):
    if df.site[i] in np.array(df.site[:i]):
        print(i)
        df.loc[i, 'site'] = '_'+df.site[i]

#%% calculate spectra and other stats
impedance = 340*1.2
spectra = []
bitweight = np.ones(df.shape[0]) * 3.5012e-3 # Pa/count--this is correct for all recordings made on modern Gems (post 2017)
bitweight[7:9] = 0.256/2**15 / (3.35/7 * 46e-6 * (1+49.7*(1/2.2+1/1))) # lucky peak in 2017
bitweight[13:15] = 0.256/2**15 / (3.35/7 * 46e-6 * (1+49.7/2.2)) # Diversion Dam 2017
bitweight[9] = 0.256/2**15 / (3.1/7 * 46e-6 * (1+49.7/2.2)) # mesa falls
df['rms'] = df['geo_mean_freq'] = df['mean_freq'] = df['med_freq'] = df['power_acoustic_W'] = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    print((i, df.site[i]))
    tr = obspy.read('../mseed/spectra_mseeds/' + df.filename[i])[0] # still in counts, need to fix this
    tr.data = tr.data * bitweight[i]
    #tr.filter('highpass', freq = df.freq_low[i], corners = 2)
    tr.filter('highpass', freq = 0.01, corners = 4)
    t1 = obspy.UTCDateTime(df.date.astype(str)[i] + ' ' + df.t1.astype(str)[i])
    t2 = obspy.UTCDateTime(df.date.astype(str)[i] + ' ' + df.t2.astype(str)[i])
    tr.trim(t1, t2)
    s = riversound.spectrum(tr)
    spectra.append(s) # save the barely-filtered spectrum to show low-freq background noise
    # recalculate the spectrum after filtering to exclude low-freq background noise
    tr.filter('highpass', freq = df.freq_low[i], corners = 2)
    s = riversound.spectrum(tr) 
    df.loc[i, 'rms'] = np.std(tr.data) 
    df.loc[i, 'power_acoustic_W'] = df['rms'][i]**2 * 2*np.pi * df['distance_m'][i]**2 / impedance
    df.loc[i, 'geo_mean_freq'] = calc_geo_mean_freq(s['freqs'], s['median'])
    df.loc[i, 'mean_freq'] = calc_mean_freq(s['freqs'], s['median'])
    df.loc[i, 'med_freq'] = calc_med_freq(s['freqs'], s['median'])
#%% plot recorded spectra (Pa^2/Hz)
## important to use median spectrum to avoid intermittent noise, e.g. vehicles at Niagara
waterfall_plot_indices = [1,2,3,4,5,6,15,16]
fig, ax = plt.subplots(2, 1, figsize = [6.5, 9.5])
for i in waterfall_plot_indices:
    ax[0].loglog(spectra[i]['freqs'], spectra[i]['median'], 
               label = df.site[i], color = df.cbcolor[i], linestyle = df.linestyle[i])
    cp = np.argmin(np.abs(df.freq_low[i] - spectra[i]['freqs']))
    ax[0].loglog(spectra[i]['freqs'][cp], spectra[i]['median'][cp], marker = 'o', color = df.cbcolor[i])

# plot the degree-3 power law for comparison
ax[0].loglog([1, 30], [3e-1, 3e-1 * 30**-3], 'k-', label = 'Degree -3 Slope')

ax[0].set_xlim([0.1, 30])
ax[0].set_title('A. Recorded Acoustic Power Spectra', loc = 'left')
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power Spectral Density (Pa$^2$/Hz)')
#%% plot source spectra (W/Hz)
## important to use median spectrum to avoid intermittent noise, e.g. vehicles at Niagara
waterfall_plot_indices = [1,2,3,4,5,6,15,16]
for i in waterfall_plot_indices:
    w = np.where(spectra[i]['freqs'] > df.freq_low[i])[0]
    ax[1].loglog(spectra[i]['freqs'][w], spectra[i]['median'][w] * df.distance_m[i]**2 * 2*np.pi/impedance, 
               label = df.site[i], color = df.cbcolor[i], linestyle = df.linestyle[i])
ax[1].set_xlim([0.1, 30])
#plt.legend()
ax[1].set_title('B. Source Acoustic Power Spectra', loc = 'left')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Source Power Spectral Density (W/Hz)')

#%%
ax[0].legend()
fig.tight_layout()
fig.savefig('../figures/WaterfallSpectra.png')
#%% Plot powers
fig, ax = plt.subplots(2, 2, figsize = [9.5, 7.5])
for i in range(df.shape[0]):
    if df.site[i][0] == '_':
        ax[0,0].loglog(df.power_hydraulic_W[(i-1):(i+1)], df.power_acoustic_W[(i-1):(i+1)], 
                       color = df.cbcolor[i], linestyle = 'solid')
    ax[0,0].loglog(df.power_hydraulic_W[i], df.power_acoustic_W[i], 
               color = df.cbcolor[i], marker = df.marker[i], linestyle = 'none', markersize=8, # default markersize=6
               label = df.site[i])
ax[0,0].loglog([3e3, 3e9], [3e-3, 3e3], color = 'gray', linestyle = '--')

ax[0,0].set_title('A. Acoustic vs. Hydraulic Power', loc = 'left')
ax[0,0].set_xlabel('Hydraulic Power (W)')
ax[0,0].set_ylabel('Acoustic Power (W)')

## the simple factor of 1e6 is consistent with the stats
scipy.stats.linregress(np.log10(df.power_hydraulic_W), np.log10(df.power_acoustic_W)) # slope is 0.9, with std err 0.08
np.mean(np.log10(df.power_acoustic_W) - np.log10(df.power_hydraulic_W)) # forcing slope 1, intercept is -5.97
## Plot frequencies
## using med_freq here. mean_freq, geo_mean_freq, and med_freq all work about the same; mean_freq 
## is higher than the others, and med_freq is easy to explain.
for i in range(df.shape[0]):
    if df.site[i][0] == '_':
        ax[0,1].loglog((df.discharge_m3s)[(i-1):(i+1)], df.ratio_acoustic_hydraulic[(i-1):(i+1)], color = df.cbcolor[i], linestyle = 'solid')
    ax[0,1].loglog((df.discharge_m3s)[i], df.ratio_acoustic_hydraulic[i], 
               color = df.cbcolor[i], marker = df.marker[i], linestyle = 'none', markersize=8, # default markersize=6
               label = df.site[i])
yticks = [2e-7, 5e-7, 1e-6, 2e-6, 5e-6]
ax[0,1].set_yticks(yticks, yticks)
ax[0,1].set_title('B. Power Ratio vs. Discharge', loc = 'left')
ax[0,1].set_xlabel('Discharge (m$^3$/s)')
ax[0,1].set_ylabel('Acoustic-Hydraulic Power Ratio')      

for i in range(df.shape[0]):
    if df.site[i][0] == '_':
        ax[1,0].loglog((df.height_m)[(i-1):(i+1)], df.med_freq[(i-1):(i+1)], color = df.cbcolor[i], linestyle = 'solid')
    ax[1,0].loglog((df.height_m)[i], df.med_freq[i], 
               color = df.cbcolor[i], marker = df.marker[i], linestyle = 'none', markersize=8, # default markersize=6
               label = df.site[i])
ax[1,0].loglog([5, 100], 20*np.array([5,100])**-0.25, color = 'gray', linestyle = '--') #degree -1/4 line
ax[1,0].set_yticks([2,5,10,20], [2,5,10,20])
ax[1,0].set_title('C. Frequency vs. Height', loc = 'left')
ax[1,0].set_xlabel('Waterfall Height (m)')
ax[1,0].set_ylabel('Median Frequency (Hz)')    

for i in range(df.shape[0]):
    if df.site[i][0] == '_':
        ax[1,1].loglog((df.discharge_m3s)[(i-1):(i+1)], df.med_freq[(i-1):(i+1)], color = df.cbcolor[i], linestyle = 'solid')
    ax[1,1].loglog((df.discharge_m3s)[i], df.med_freq[i], 
               color = df.cbcolor[i], marker = df.marker[i], linestyle = 'none', markersize=8, # default markersize=6
               label = df.site[i])
ax[1,1].loglog([0.25, 4000], 10**1.16*np.array([0.25,4000])**-0.1941, color = 'gray', linestyle = '--') #degree -1/4 line

ax[1,1].set_yticks([2,5,10,20], [2,5,10,20])
ax[1,1].set_title('D. Frequency vs. Discharge', loc = 'left')
ax[1,1].set_xlabel('Discharge (m$^3$/s)')
ax[1,1].set_ylabel('Median Frequency (Hz)')

scipy.stats.linregress(np.log10(df.power_hydraulic_W), np.log10(df.med_freq)) # slope is 0.9, with std err 0.08
scipy.stats.linregress(np.log10(df.discharge_m3s), np.log10(df.med_freq)) # slope is 0.9, with std err 0.08
scipy.stats.linregress(np.log10(df.height_m), np.log10(df.med_freq)) # slope is 0.9, with std err 0.08

ax[1,0].legend(
    ncol=2,
    markerscale=0.7,
    handlelength=1,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.2
    )

fig.tight_layout()
fig.savefig('../figures/WaterfallStats.png')
