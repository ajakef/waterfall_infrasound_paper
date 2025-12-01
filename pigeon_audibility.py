import numpy as np
import pandas as pd
import obspy
import gemlog
import os
import matplotlib.pyplot as plt
import riversound
os.chdir('/home/jake/Dropbox/StreamAcoustics/waterfall_paper/code')
#import sys; sys.path.append('code')
from waterfall_functions import *

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
    if df.site[i] in np.array(df.site[:i]):
        df.loc[i, 'site'] = '_'+df.site[i]

#%% plot Niagara spectrum vs pigeon audiogram
## the calculated spectra are one-sided, so no need to double it
#xlim = [0.2, 30]
xlim = freq_range
#fig, ax = plt.subplots(4,2, figsize = (9.5, 6.5))
fig = plt.figure(figsize = (6.5*1.5, 9.5*1.5))
gs = fig.add_gridspec(5,2)
ax1 = []
ax2 = []

for r, i in enumerate([5, 3, 12]): # 5 Niagara high, 3 Palouse high, 12 Tahquamenon high
    ax1.append(fig.add_subplot(gs[r,0]))
    ax1[r].loglog(freqs, spectra[i]['median'], 'k-')
    ax1[r].set_xlim(xlim)
    ax1[r].set_ylabel('Observed PSD (Pa$^2$/Hz)')
    ax1[r].set_yticks(10.0**np.array([-1,-3,-5,-7]))
    k = np.sqrt(np.max(audiogram[freqs>=(0.8*freq_range[0])])/np.min(audiogram[freqs<=freq_range[1]]))
    #ylim = np.sqrt(np.max(spectra[i]['median']) * np.min(spectra[i]['median'])) * np.array([1/k, k])
    ylim = sorted(0.25/np.array([1, k**2]))
    ax1[r].set_ylim(ylim)
    ax1[r].set_title(f'{"ACE"[r]}. {df.site[i]} Power Spectrum', loc = 'left')
    ax1[r].text(xlim[0]*2, 6*ylim[0], f'{np.round(np.sum(spectra[i]["median"]) * dfreq, 4)} Pa$^2$ at {int(np.round(df.distance_m[i]))} m', va = 'bottom')
    ax1[r].text(xlim[0]*2, 1*ylim[0], f'Hydr. Power: {int(np.round(df.power_hydraulic_W[i]/1e6))} MW', va = 'bottom')
    
    right_y_axis = ax1[r].twinx()
    l2, = right_y_axis.loglog(freqs, audiogram, label = 'Audiogram', color = 'maroon')
    right_y_axis.set_ylabel('Audiogram (Pa$^2$)', color = 'maroon')
    right_y_axis.tick_params(colors = 'maroon')
    ylim = ax1[r].get_ylim()/ax1[r].get_ylim()[0] * np.min(audiogram[freqs<=freq_range[1]])
    right_y_axis.set_ylim(ylim)
    right_y_axis.set_yticks(10.0**np.array([1, -1,-3,-5]))

    #distance = df.distance_m * np.sum((spectra[i]['median']/audiogram)[w]) * dfreq
    dist_scale_factor = (df.pigeon_audibility_distance_m[i]/df.distance_m[i])**2
    ax2.append(fig.add_subplot(gs[r,1]))
    ax2[r].plot(freqs, spectra[i]['median'] / audiogram / dist_scale_factor)
    ax2[r].set_xlim((0, xlim[1]))
    ax2[r].set_ylim((0, 0.16))
    ax2[r].set_yticks(0.05*np.arange(4))
    ax2[r].set_ylabel('PSD/Audiogram (1/Hz)')
    ax2[r].set_title(f'{"BDF"[r]}. {df.site[i]} Audibility ({int(np.round(df.pigeon_audibility_distance_m[i]))} m)', loc = 'left')
    

ax1[2].set_xlabel('Frequency (Hz)')
ax2[2].set_xlabel('Frequency (Hz)')

## logic: if integral(power/audiogram * df) > 1, the sound is audible. 
## This integral is around 130 by about 27 Hz. So, it *should* be audible out to sqrt(130)*sensor_distace
## This comes out to 8.3 km.

#fig, ax = plt.subplots(1,1, figsize = (9.5, 6.5))
#ax.loglog(df.power_hydraulic_W, df.pigeon_audibility_distance_m, 'k.')
ax = fig.add_subplot(gs[3:,:])
for i in range(df.shape[0]):
    if df.site[i][0] == '_':
        ax.loglog(df.power_hydraulic_W[(i-1):(i+1)], df.pigeon_audibility_distance_m[(i-1):(i+1)], 
                       color = df.cbcolor[i], linestyle = 'solid')
    ax.loglog(df.power_hydraulic_W[i], df.pigeon_audibility_distance_m[i], 
               color = df.cbcolor[i], marker = df.marker[i], linestyle = 'none', markersize=8, # default markersize=6
               label = df.site[i])

lr = scipy.stats.linregress(np.log10(df.power_hydraulic_W), np.log10(df.pigeon_audibility_distance_m)) # 0.34 power law

ax.loglog([1e3, 1e10], 10**lr.intercept*np.array([1e3, 1e10])**lr.slope, color = 'gray', linestyle = '--')
ax.text(1e8, 1e3, f'y={10**lr.intercept:0.2f}$\\times 10^{{{lr.slope:0.2f}x}}$, $r^2$={lr.rvalue**2:0.2f}')
ax.legend(
    ncol=2,
    markerscale=0.7,
    handlelength=1,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.2
    )

ax.set_xlim([5e3, 5e9])
ax.set_xlabel('Hydraulic Power (W)')
ax.set_ylabel('Pigeon Audibility Distance (m)')
ax.set_title('G. Audibility Distance vs. Waterfall Hydraulic Power', loc = 'left')

fig.tight_layout()
fig.savefig('figures/PigeonAudibility.png')
