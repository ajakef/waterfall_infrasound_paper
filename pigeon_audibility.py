#!/usr/bin/env python3

#%% imports
import numpy as np
import pandas as pd
import obspy
import gemlog
import os
import matplotlib.pyplot as plt
import riversound
import sys; sys.path.append('code')
from waterfall_functions import *
os.chdir('/home/jake/Dropbox/StreamAcoustics/waterfall_paper')

#%% calculate waterfall spectra
impedance = 340*1.2
spectra = []
df = pd.read_excel("other_data/waterfall_summary.ods", engine="odf", skiprows=1).convert_dtypes()

bitweight = np.ones(df.shape[0]) * 3.5012e-3 # Pa/count--this is correct for all recordings made on modern Gems (post 2017)
bitweight[7:9] = 0.256/2**15 / (3.35/7 * 46e-6 * (1+49.7*(1/2.2+1/1))) # lucky peak in 2017
bitweight[13:15] = 0.256/2**15 / (3.35/7 * 46e-6 * (1+49.7/2.2)) # Diversion Dam 2017
bitweight[9] = 0.256/2**15 / (3.1/7 * 46e-6 * (1+49.7/2.2)) # mesa falls
df['rms'] = df['geo_mean_freq'] = df['mean_freq'] = df['med_freq'] = df['power_acoustic_W'] = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    print((i, df.site[i]))
    tr = obspy.read('mseed/spectra_mseeds/' + df.filename[i])[0] # still in counts, need to fix this
    tr.data = tr.data * bitweight[i]
    tr.filter('highpass', freq = df.freq_low[i], corners = 2)
    t1 = obspy.UTCDateTime(df.date.astype(str)[i] + ' ' + df.t1.astype(str)[i])
    t2 = obspy.UTCDateTime(df.date.astype(str)[i] + ' ' + df.t2.astype(str)[i])
    tr.trim(t1, t2)
    s = riversound.spectrum(tr)
    df.loc[i, 'rms'] = np.std(tr.data) 
    df.loc[i, 'power_acoustic_W'] = df['rms'][i]**2 * 2*np.pi * df['distance_m'][i]**2 / impedance
    df.loc[i, 'geo_mean_freq'] = calc_geo_mean_freq(s['freqs'], s['median'])
    df.loc[i, 'mean_freq'] = calc_mean_freq(s['freqs'], s['median'])
    df.loc[i, 'med_freq'] = calc_med_freq(s['freqs'], s['median'])
    spectra.append(s)

#%% calculate audibility for all waterfalls
# This assumes that acoustic power at all low frequencies can be scaled by the audiogram and
# summed to determine audibility. So, it is a generous estimate of how far away a waterfall
# can be heard.
audiogram_df = pd.read_csv('other_data/PigeonAudiogram.csv', names = ['freq', 'dB'])
audiogram_df['Pa2'] = 10**(audiogram_df['dB']/10) * (20e-6)**2
audiogram = (20e-6)**2 * 10**(np.interp(np.log10(spectra[0]['freqs']), np.log10(audiogram_df.freq), audiogram_df.dB)/10)

freqs = spectra[0]['freqs']
dfreq = np.diff(freqs)[0]
freq_range = [0.5, 30] # min and max frequencies to consider for audibility
w = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]

df['pigeon_audibility_distance_m'] = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    #w = np.where(spectra[i]['median'] > audiogram)[0]
    audibility = np.sum((spectra[i]['median']/audiogram)[w]) * 1/10.24
    df.loc[i, 'pigeon_audibility_distance_m'] = df.distance_m[i] * np.sqrt(audibility)
    
#%% plot Niagara spectrum vs pigeon audiogram
## the calculated spectra are one-sided, so no need to double it
#xlim = [0.2, 30]
xlim = freq_range
fig, ax = plt.subplots(3,2, figsize = (9.5, 6.5))

for r, i in enumerate([5, 3, 12]): # 5 Niagara high, 3 Palouse high, 12 Tahquamenon high
    ax[r,0].loglog(freqs, spectra[i]['median'], 'k-')
    ax[r,0].set_xlim(xlim)
    ax[r,0].set_ylabel('Observed PSD (Pa$^2$/Hz)')
    k = np.sqrt(np.max(audiogram)/np.min(audiogram))
    ylim = np.sqrt(np.max(spectra[i]['median']) * np.min(spectra[i]['median'])) * np.array([1/k, k])
    ax[r,0].set_ylim(ylim)
    ax[r,0].set_title(f'{"ACE"[r]}. {df.site[i]} Power Spectrum', loc = 'left')
    ax[r,0].text(xlim[0]*1.2, 12*ylim[0], f'Total: {np.round(np.sum(spectra[i]["median"]) * dfreq, 4)} Pa$^2$ at {int(np.round(df.distance_m[i]))} m', va = 'bottom')
    ax[r,0].text(xlim[0]*1.2, 2*ylim[0], f'Hydr. Power: {int(np.round(df.power_hydraulic_W[i])):.2g} W', va = 'bottom')
    
    right_y_axis = ax[r,0].twinx()
    l2, = right_y_axis.loglog(freqs, audiogram, label = 'Audiogram', color = 'maroon')
    right_y_axis.set_ylabel('Audiogram (Pa$^2$)', color = 'maroon')
    right_y_axis.tick_params(colors = 'maroon')
    ylim = ax[r,0].get_ylim()/ax[r,0].get_ylim()[0] * np.min(audiogram)
    right_y_axis.set_ylim(ylim)

    #distance = df.distance_m * np.sum((spectra[i]['median']/audiogram)[w]) * dfreq
    dist_scale_factor = (df.pigeon_audibility_distance_m[i]/df.distance_m[i])**2
    ax[r,1].plot(freqs, spectra[i]['median'] / audiogram / dist_scale_factor)
    ax[r,1].set_xlim((0, xlim[1]))
    ax[r,1].set_ylabel('PSD/Audiogram (1/Hz)')
    ax[r,1].set_title(f'{"BDF"[r]}. {df.site[i]} Audibility ({int(np.round(df.pigeon_audibility_distance_m[i]))} m)', loc = 'left')
    

ax[-1, 0].set_xlabel('Frequency (Hz)')
ax[-1, 1].set_xlabel('Frequency (Hz)')
fig.tight_layout()
fig.savefig('figures/PigeonAudibility.png')

## logic: if integral(power/audiogram * df) > 1, the sound is audible. 
## This integral is around 130 by about 27 Hz. So, it *should* be audible out to sqrt(130)*sensor_distace
## This comes out to 8.3 km.
