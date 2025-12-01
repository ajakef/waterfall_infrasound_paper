import matplotlib.pyplot as plt
import numpy as np
import gemlog
import glob
import pandas as pd
import obspy
import os
import riversound
import scipy
import os
os.chdir('/home/jake/Dropbox/StreamAcoustics/waterfall_paper/code')
from waterfall_functions import *


#%% Palouse Falls: read discharge and infrasound
df = pd.read_csv("../data/PalouseFallsDischarge.txt", sep = '\t', skiprows=25, names = ['USGS', 'site', 'datetime', 'tz', 'discharge', 'valid'])
df['t'] = pd.to_datetime(df.datetime, utc = True) + pd.to_timedelta(7, unit='h') # USGS provides data in PDT
df['discharge'] /= 35.315 # convert to m3/sec
#plt.plot(df.t, df.discharge)

st = obspy.Stream()
for fn in sorted(glob.glob('../data/waterfall_recordings/PalouseFalls/2024_infrasound_miniSEED/2024-0*..122..HDF.mseed')):
    st += obspy.read(fn)
st = st.merge(method = 1, fill_value = 'interpolate', interpolation_samples = 100)

assert len(st) > 0, "len(st) = 0; probably the path is wrong or the infrasound data missing"

#%% process Palouse Falls data
sg = []
rms = []
medfreqs = []
t1 = obspy.UTCDateTime(2024,5,15,10,0,0)
t2 = obspy.UTCDateTime(2024,6,28,11,0,0)
t = t1
while t < t2:
    print(t)
    tr = st.slice(t, t+3600).merge()[0]
    tr.filter('highpass', freq = 1, corners = 4)
    tr.data = tr.data * 3.5012e-3 
    print(tr)
    s = riversound.spectrum(tr)
    sg.append(s['median'])
    rms.append(tr.std())
    medfreqs.append(calc_med_freq(s['freqs'], s['median']))
    t += 86400

rms = np.array(rms)
medfreqs = np.array(medfreqs)

#%% plot Palouse Falls data
from matplotlib.ticker import FixedLocator, FuncFormatter

impedance = 340 * 1.2
distance = 320.9
t = pd.Timestamp(str(t1)) + pd.to_timedelta(np.arange(len(sg)), unit='D') + pd.to_timedelta(0.5, unit='h')

plt.close('all')
fig, ax = plt.subplots(4,1, figsize = [13, 9.5])

# linear y-axis
#l1, = ax[0].plot(df.t, df.discharge)
#right_y_axis = ax[0].twinx()
#l2, = right_y_axis.plot(t, rms**2 * 2*np.pi*distance**2/impedance, color = 'orange')
#right_y_axis.set_ylabel('Acoustic Power (W)')

# log y-axis
l1, = ax[0].semilogy(df.t, df.discharge, label = 'Discharge')
ax[0].yaxis.set_major_locator(FixedLocator([100,200,300]))
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
right_y_axis = ax[0].twinx()
l2, = right_y_axis.semilogy(t, rms**2 * 2*np.pi*distance**2/impedance, 'k.', label = 'Acoustic Power')
right_y_axis.set_ylabel('Acoustic Power (W)')
right_y_axis.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
right_y_axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

ax[0].set_xlim([pd.Timestamp('20240515', tz = 'UTC'), pd.Timestamp('20240629', tz = 'UTC')]) 
ax[0].set_ylabel(r'Discharge (m$^3$/s)')
ax[0].set_title('A. Palouse Falls Discharge and Acoustic Power', loc = 'left')
ax[0].legend([l1, l2], ['Discharge', 'Acoustic Power'], loc = 'lower left')

riversound.image(np.log10(np.array(sg)), t, s['freqs'], log_y = True, qmin = 0.3, ax = ax[1])
ax[1].plot(t, np.log10(medfreqs), color = 'black')
ax[1].set_ylim([np.log10(2), np.log10(25)])
ax[1].set_yticks(np.log10([2,5,10,20]), [2,5,10,20])

ax[1].set_ylabel('Frequency (Hz)')
ax[1].set_xlabel('Date')
ax[1].set_xlim([pd.Timestamp('20240515', tz = 'UTC'), pd.Timestamp('20240629', tz = 'UTC')]) 
ax[1].set_title('B. Palouse Falls Spectrogram', loc = 'left')

fig.tight_layout()


#%% Lucky Peak
st = obspy.Stream()
for fn in sorted(glob.glob('../data/waterfall_recordings/LuckyPeak/2017*/2017*')): # 013, limited 010
    print(fn)
    st += obspy.read(fn)

assert len(st) > 0, "len(st) = 0; probably the path is wrong or the infrasound data missing"

#%% Lucky Peak process data
sg = []
rms = []
medfreqs = []
t1 = obspy.UTCDateTime(2017,5,18,9,0,0)
t2 = obspy.UTCDateTime(2017,6,20,11,0,0)
bitweight = 0.256/2**15 / (3.35/7 * 46e-6 * (1+49.7*(1/2.2+1/1)))
t = t1
while t < t2:
    print(t)
    st_tmp = st.select(station='013').slice(t, t+2*3600).merge(method = 1, fill_value = 'interpolate', interpolation_samples = 100)
    print(st_tmp)
    if (len(st_tmp) == 0) or (len(st_tmp[0]) < 90000): # skip if inadequate data
        print(f'skip {t}')
        medfreqs.append(np.nan)
        sg.append(np.nan + np.zeros(513))
        rms.append(np.nan)
        t+=86400
        continue
    tr = st_tmp[0]
    tr.filter('highpass', freq = 1, corners = 4)
    tr.data = tr.data * bitweight
    s = riversound.spectrum(tr)
    medfreqs.append(calc_med_freq(s['freqs'], s['median']))
    sg.append(s['median'])
    # calculate power from med spec (safer in case of clips or invalid data)
    rms.append(np.sqrt(2*np.sum(s['median']) * np.diff(s['freqs'])[0]))
    t += 86400


rms = np.array(rms)
medfreqs = np.array(medfreqs)
#%% Lucky Peak discharge
df = pd.read_csv("../data/BoiseRiverDischarge.csv", skiprows=2)#.iloc[:(365*24*4),:]
df['t'] = pd.to_datetime(df.DATETIME) + pd.to_timedelta(6, unit='h') # USGS provides data in PDT
df['Over_Div_Dam'] = pd.to_numeric(df.Over_Div_Dam, errors = 'coerce')/35.315 # convert to m3/sec
df['LUC_QR'] = pd.to_numeric(df.LUC_QR, errors = 'coerce')/35.315 # convert to m3/sec
df['LUC_XQ'] = pd.to_numeric(df.LUC_XQ, errors = 'coerce')/35.315 # convert to m3/sec, total
df['LUC_QS'] = pd.to_numeric(df.LUC_QS, errors = 'coerce')/35.315 # convert to m3/sec, rooster tail


df['discharge'] = df['LUC_QR']


#%% plot Lucky Peak

from matplotlib.ticker import FixedLocator, FuncFormatter

impedance = 340 * 1.2
distance = 145
t = pd.Timestamp(str(t1)) + pd.to_timedelta(np.arange(len(sg)), unit='D') + pd.to_timedelta(0.5, unit='h')

F = 5

# linear y-axis
#l1, = ax[2].plot(df.t, df.discharge)
#right_y_axis = ax[2].twinx()
#l2, = right_y_axis.plot(t, rms**2 * 2*np.pi*distance**2/impedance, color = 'orange')
#right_y_axis.set_ylabel('Acoustic Power (W)')

# log y-axis
l1, = ax[2].semilogy(df.t[df.discharge > 10], df.discharge[df.discharge > 10], label = 'Discharge')
#ax[2].yaxis.set_major_locator(FixedLocator([100,200,300]))
#ax[2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
right_y_axis = ax[2].twinx()
power = rms**2 * 2*np.pi*distance**2/impedance
w = power>5
l2, = right_y_axis.semilogy(t[w], power[w], 'k.', label = 'Acoustic Power')
right_y_axis.set_ylabel('Acoustic Power (W)')
#right_y_axis.set_ylim([300/F, 300])
#right_y_axis.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
#right_y_axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax[2].set_xlim([pd.Timestamp('20170518', tz = 'UTC'), pd.Timestamp('20170621', tz = 'UTC')]) 
ax[2].set_ylim([200/F, 180])
ax[2].set_ylabel(r'Discharge (m$^3$/s)')
ax[2].set_title('C. Lucky Peak Discharge and Acoustic Power', loc = 'left')
ax[2].legend([l1, l2], ['Discharge', 'Acoustic Power'], loc = 'lower left')


riversound.image(np.log10(np.array(sg)[w,:]), t[w], s['freqs'], log_y = True, qmin = 0.3, ax = ax[3])
ax[3].plot(t[w], np.log10(medfreqs[w]), color = 'black')
ax[3].set_ylim([np.log10(2), np.log10(25)])
ax[3].set_yticks(np.log10([2,5,10,20]), [2,5,10,20])

ax[3].set_ylabel('Frequency (Hz)')
ax[3].set_xlabel('Date')
ax[3].set_xlim([pd.Timestamp('20170518', tz = 'UTC'), pd.Timestamp('20170620', tz = 'UTC')]) 
ax[3].set_title('D. Lucky Peak Spectrogram', loc = 'left')

fig.tight_layout()
fig.savefig('../figures/WaterfallsOverTime.png')
