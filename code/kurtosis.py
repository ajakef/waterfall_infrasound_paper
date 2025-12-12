#%% imports
import numpy as np
import obspy
import matplotlib.pyplot as plt
import riversound
from scipy.stats import kurtosis
from scipy.signal import detrend
import obspy.signal.cross_correlation



#%% scaling functions for plot
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
#%% read and pre-process data
st = obspy.read('../data/kurtosis_mseeds/2024-05-20T14*2..HDF.mseed')
st.sort() # 062, the noisy one, comes first
t1 = obspy.UTCDateTime('2024-05-20T19:00')
t2 = obspy.UTCDateTime('2024-05-20T19:05')
st = st.detrend('linear')
st.filter('highpass', freq = 1)
st.trim(t1,t2)
#df = pd.read_excel("../data/waterfall_summary.ods", engine="odf", skiprows=1).convert_dtypes()
for tr in st: 
    tr.data = tr.data * 3.5012e-3
#%% calculate signal stats
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
    output_dict['sg'] = riversound.pgram(st_tmp[0], 0.01)['spectrum']
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
t_tr = np.arange(len(st_plot[0].data))*0.01
xmin = t_tr.min()
xmax = t_tr.max()
freq = 1/win_length_sec * np.arange(sg.shape[1])

#%% plot results
plt.close('all')
fig = plt.figure(figsize=(8.5, 5), constrained_layout=True)  

# 4 rows, 3 columns. width ratios 20:1:20
gs = fig.add_gridspec(
    nrows=4, ncols=3,
    width_ratios=[20,1,20],
    hspace=0.03,      # tighten vertical spacing
    wspace=0.03       # tighten horizontal spacing
)

# Left column: main small plots A-D
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0], sharex=axA)   # share x with A so x axes line up
axC = fig.add_subplot(gs[2, 0], sharex=axA)
axD = fig.add_subplot(gs[3, 0], sharex=axA)

# Middle column: narrow panels for colorbars (one per row)
caxA = fig.add_subplot(gs[0, 1])
caxB = fig.add_subplot(gs[1, 1])
caxC = fig.add_subplot(gs[2, 1])
caxD = fig.add_subplot(gs[3, 1])

# Turn off the "colorbar" unused colorbar axes 
caxA.axis('off')
caxC.axis('off')

# Right column: single panel spanning all rows (panel E)
axE = fig.add_subplot(gs[:, 2])



# panel A
axA.plot(t_tr, norm(st_plot[1].data, st_plot[0].data.max()))
axA.plot(t_tr, norm(st_plot[0].data, st_plot[0].data.max()) + 1)
axA.set_xlim(xmin, xmax)
axA.set_ylabel('sensor #')
axA.set_yticks([1,0], [1,2])
axA.set_xticks([])
axA.set_title('a. Infrasound Time Series', loc = 'left')

# panel B
w = freq < 40
im = riversound.image(np.log(clip_sg(sg[:,w])), t_mid, freq[w], log_y=True, ax=axB)
axB.set_xlim(xmin, xmax)
axB.set_ylim(np.log10([0.5, 30]))
axB.set_xticks([])
axB.set_ylabel("Frequency (Hz)")
cbar = fig.colorbar(im, cax=caxB, ticks = [])
cbar.set_label('PSD (arb. units)')
axB.set_title('b. Spectrogram (sensor 1)', loc='left')

# panel C
axC.semilogy(t_mid, k1+3)
axC.semilogy(t_mid, k0+3)
axC.axhline(2.5, 0,1, color = 'black', linestyle = '--')
axC.axhline(3.5, 0,1, color = 'black', linestyle = '--')
axC.set_xlim(xmin, xmax)
axC.set_xticks([])
axC.set_yticks([3,5,10,20],[3,5,10,20])
axC.set_ylabel('Kurtosis')
axC.set_title('c. Window Kurtosis', loc = 'left')
 
# panel D
t_lag = 0.01 * (np.arange(xc.shape[1]) - (xc.shape[1]-1)/2)
im2 = riversound.image(xc, t_mid, t_lag, ax=axD, zmin = -0.35, zmax = 1)
axD.set_xlim(xmin, xmax)
axD.set_ylabel('Lag (s)')
axD.set_title('d. Correlogram', loc='left')
axD.set_xlabel('Time (s)')
cbar = fig.colorbar(im2, cax=caxD, ticks = [-0.5, 0, 0.5, 1])
cbar.set_label('Corr. Coef.')


# panel E
linewidth_list = [3, 2, 2, 1]
color_list = ['black', 'dimgray', 'darkred', 'orange']
linestyle_list = ['solid', 'solid', 'solid', 'solid']
threshold_list = [np.inf, 2.5, 0.5, 0.1]
# plot lines for each kurtosis threshold
for threshold, color, linestyle, linewidth in zip(threshold_list, color_list, linestyle_list, linewidth_list):
    sg_sub = sg[np.abs(k0)<threshold,:]
    axE.loglog(freq, sg_sub.mean(0), color = color, linewidth = linewidth, linestyle = linestyle)
    print(sg_sub.shape[0])

axE.set_xlim(0.5, 30)
axE.set_ylim(-1.5 + dB(sg_sub.mean(0)[(freq > 5) & (freq < 40)].min()), 1.5 + dB(sg.mean(0).max()))
axE.legend(['all, n = %s' % sg.shape[0]] + ['|k - 3| < %0.1f, n = %i' % (threshold, np.sum(k0 < threshold)) for threshold in threshold_list[1:]])
axE.set_xlabel('Frequency (Hz)')
axE.set_ylabel('Power Spectral Density (Pa$^2$/Hz)', labelpad = -4) # labelpad to make it neater
axE.set_title('e. Welch spectra (sensor 1)', loc = 'left')

# don't use fig.tight_layout(); it doesn't work with this complicated layout
fig.savefig('../figures/figure_kurtosis_xc_palouse.png')
