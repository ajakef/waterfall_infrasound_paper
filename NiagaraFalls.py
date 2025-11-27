#%% imports
import numpy as np
import rasterio
import pandas as pd
import obspy
import gemlog
import os
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
os.chdir('/home/jake/Dropbox/StreamAcoustics/waterfall_paper')
#%% definitions
df = pd.read_excel("other_data/waterfall_summary.ods", engine="odf", skiprows=1).convert_dtypes()
impedance = 340*1.2
lon_HSF, lat_HSF = -79.07589, 43.07797
lon_AF, lat_AF = -79.069831, 43.084487
def model_NF_power(lon, lat, power_total = df.power_hydraulic_W[6] * 1e-6):
    # [5] is high flow in tourist hours, [6] is non-tourist hours 
    r_HSF = np.sqrt(((lon - lon_HSF)*40e6/360*np.cos(43.08*np.pi/180))**2 + ((lat-lat_HSF)*40e6/360)**2)
    r_AF = np.sqrt(((lon - lon_AF)*40e6/360*np.cos(43.08*np.pi/180))**2 + ((lat-lat_AF)*40e6/360)**2)
    power_HSF = power_total * 0.9 # discharge ratio: https://www.niagarafallsstatepark.com/park-information/amazing-niagara-facts/
    power_AF = power_total * 0.1
    p_Pa = np.sqrt(impedance/(2*np.pi) * (power_HSF / r_HSF**2 + power_AF / r_AF**2))
    p_dB = 20*np.log10(p_Pa/20e-6)
    return p_dB

if True:
    # Good low-flow interval
    t1 = obspy.UTCDateTime('2024-09-19 04:30')
    t2 = obspy.UTCDateTime('2024-09-19 05:08')
    power_total = 1e-6 * df.power_hydraulic_W[6]
else:
    # Good but short tourist hours
    t1 = obspy.UTCDateTime('2024-09-19 12:14:30')
    t2 = obspy.UTCDateTime('2024-09-19 12:19:45')
    power_total = 1e-6 * df.power_hydraulic_W[5]
    
#%% Make the Niagara Falls regional satellite photo
# download Sentinel-2 image (log in to copernicus.eu)
# True color, L2A (not L1C), 32-bit float, high res, TIFF

# Read Sentinel-2 image
tif_path = "other_data/aerial_photos/2025-07-04-00_00_2025-07-04-23_59_Sentinel-2_L2A_True_color_v2.tiff"

with rasterio.open(tif_path) as src:
    img = src.read([1, 2, 3])  # read R,G,B bands
    extent = plotting_extent(src) # have to use the extent in the source image, or trim it by indexing

# Rearrange for matplotlib (H, W, 3)
img = np.transpose(img, (1, 2, 0))

fig, axes = plt.subplots(2, 1, figsize=(6.5, 9.5))
ax = axes[0]

ax.imshow(img, extent=extent, origin='upper')

lon, lat = np.meshgrid(np.arange(extent[0], extent[1], 0.0001), np.arange(extent[2], extent[3], 0.0001))

p_dB = model_NF_power(lon, lat, power_total)

# Overlay contours
CS = ax.contour(lon, lat, p_dB, colors='white', linewidths=1, levels = np.arange(55, 86, 5))
ax.clabel(CS, inline=True, fontsize=12, fmt="%.0f dB")

ax.plot(-78.966 + np.arange(2) * 1000/ (40e6/360 * np.cos(np.deg2rad(43.08))), [43.055, 43.055], 'w-', linewidth=4)
ax.text(-78.967, 43.056, '1 km', color = 'white', size = 12, va = 'bottom')
ax.set_xlabel("Longitude")
ax.set_aspect(1 / np.cos(np.deg2rad(43.08)))

ax.set_ylabel("Latitude")
ax.set_title("A. Modeled Infrasound from Niagara Falls", loc = 'left')

#%% get data from various regional sensors and plot them
# 247 Goat Island
# 251 DeVeaux
# 232 ArtPark
# 255 house

gps_dir = '/home/jake/Dropbox/StreamAcoustics/NiagaraFalls/2024-09-24_Download/gps'
SN_list = sorted(['247', '251', '232', '255']) # Goat Island, DeVeaux Woods, ArtPark, House

df_gps = gemlog.summarize_gps(gps_dir, t1 = t1-3600, t2 = t2+3600, include_SN = SN_list)

#%% calculate infrasound dB at the selected regional sites
dB = np.zeros(len(SN_list))
for i, SN in enumerate(df_gps.SN):
    tr = obspy.read(f'mseed/NiagaraFalls/{t1.strftime("%Y-%m-%d")}*{SN}*')[0]
    tr.filter('highpass', freq = 0.5, corners = 4)
    tr.trim(t1, t2)
    dB[i] = 20*np.log10(tr.std() * 3.5012e-3/20e-6)
   
dB_diff = dB - model_NF_power(df_gps.lon, df_gps.lat, power_total)   
#%% Add Hyde Park data (optional--it's a short dataset, daytime only, with noise and amplitude uncertainty)
## 062 actually appears to be reading low at high freqs
if False:
    df_gps = pd.concat([df_gps, gemlog.summarize_gps('../NiagaraFalls/2024-08-15_NiagaraFalls/2024-08-15_HydePark/gps', include_SN = '062')]).reset_index()
    tr = obspy.read(f'/home/jake/Dropbox/StreamAcoustics/NiagaraFalls/2024-08-15_NiagaraFalls/2024-08-15_HydePark/mseed/*062*')[0]
    tr.filter('highpass', freq = 0.5, corners = 4)
    tr.trim(obspy.UTCDateTime('2024-08-15 13:02:26'), obspy.UTCDateTime('2024-08-15 13:08:07')) ## high flow
    #tr.plot()
    dB_diff = np.append(dB_diff, 20*np.log10(tr.std() * 3.5012e-3/20e-6) - model_NF_power(df_gps.iloc[-1, :].lon, df_gps.iloc[-1, :].lat, df.power_hydraulic_W[5]*1e-6))
#%% plot the regional dB differences
norm = TwoSlopeNorm(vmin=-np.max(np.abs(dB_diff)), vcenter=0, vmax=np.max(np.abs(dB_diff)))
sc = ax.scatter(df_gps.lon, df_gps.lat, c = dB_diff, norm = norm, cmap = 'bwr', s = 80, edgecolor = 'k')
cbar = plt.colorbar(sc, ax=ax, shrink = 0.333)
cbar.set_label('Observed - Modeled (dB)')


#%% Inset of falls: read local aerial photo
# https://open.niagarafalls.ca/datasets/ad640c1367934f89b4ebcfe3256dd02c/explore?location=43.079244%2C-79.081029%2C15.00
# "Contains information licensed under the Open Government Licence – Niagara Falls (Ontario, Canada)."
# City of Niagara Falls (Ontario) Open Data
ax = axes[1]
tif_path = 'other_data/aerial_photos/Niagara Falls 2018 Ortho Imagery.jpg'
with rasterio.open(tif_path) as src:
    img = src.read([1, 2, 3])  # read R,G,B bands

# Unlike the regional Sentinel-2 image, this aerial photo doesn't include an extent.
# So, calculate it manually using the edges of the waterfalls as reference points.
[x1, y1] = [119, 296] 
[lon1, lat1] = [-79.07810, 43.07895] # SW of Horseshoe Falls
[x2, y2] = [247, 163] 
[lon2, lat2] = [-79.06876, 43.08570] # NE of American Falls

extent = [
        (lon2-lon1)/(x2-x1) * (0-x1) + lon1,
        (lon2-lon1)/(x2-x1) * (400-x1) + lon1,     
        (lat2-lat1)/(y2-y1) * (400-y1) + lat1,
        (lat2-lat1)/(y2-y1) * (0-y1) + lat1   
    ]

img = np.transpose(img, (1, 2, 0)) # Rearrange for matplotlib (H, W, 3)
ax.imshow(img, extent=extent, origin='upper')

# model infrasound dB over the map
lon, lat = np.meshgrid(np.arange(extent[0], extent[1], 0.0001), np.arange(extent[2], extent[3], 0.0001))
p_dB = model_NF_power(lon, lat, power_total = 1e-6 * df.power_hydraulic_W[5])

# Overlay contours on map
CS = ax.contour(lon, lat, p_dB, colors='white', linewidths=1, levels = np.arange(55, 96, 5))
ax.clabel(CS, inline=True, fontsize=12, fmt="%.0f dB")

# add the scale bar
ax.plot(-79.063 + np.arange(2) * 200/ (40e6/360 * np.cos(np.deg2rad(43.08))), [43.0743, 43.0743], 'w-', linewidth=4)
ax.text(-79.063, 43.0745, '200 m', color = 'white', size = 12, va = 'bottom')

# add labels for Horseshoe Falls and American Falls
ax.text(lon_AF, lat_AF, 'AF', va = 'center', ha = 'center', size = 'x-large')
ax.text(lon_HSF, lat_HSF, 'HF', va = 'center', ha = 'center', size = 'x-large')

ax.set_xlabel("Longitude")
ax.set_aspect(1 / np.cos(np.deg2rad(43.08)))
ax.set_ylabel("Latitude")
ax.set_title("B. Modeled Infrasound from Niagara Falls (detail)", loc = 'left')

#%% Calculate lats and lons for Aug 14 infrasound survey stops
# 268 2024-08-14 21:22 UTC SW Edge of American Falls (Luna Island) ~1 Pa RMS
# 268 2024-08-14 21:31 UTC 1/3 of the way from AF to HF ~0.84 Pa RMS
# 268 2024-08-14 21:41 UTC 1/2 of the way from AF to HF ~1 Pa RMS
# 268 2024-08-14 21:45 UTC 154 m NE of HF ~1.2 Pa RMS
# 268 2024-08-14 21:54 UTC NE Edge of Horseshoe Falls ~2 Pa RMS
# 268 2024-08-14 23:09 UTC Maid of the Mist Horseshoe Falls ~3 Pa RMS
# 268 2024-08-14 23:22 UTC 100 m NE of American Falls base ~0.9 Pa RMS
# 266 2024-08-15 12:48 UTC Hyde Park
# 266 2024-08-15 14:48:30-15:00:30 UTC falls view park ~1 Pa RMS
# 266 2024-08-15 16:26:30-16:19:30 UTC tunnel outlet ~3 Pa RMS
# 266 2024-08-15 17:20:00-17:21:40 UTC Canada parking lot

df_gps = gemlog.read_gps('other_data/NiagaraFalls_GPS', SN = '268')
tr = obspy.read('mseed/NiagaraFalls/2024-08-14T21_01_18..268..HDF.mseed')[0]
tr.filter('highpass', freq = 0.5)
t = np.array([obspy.UTCDateTime(t) for t in df_gps.t])

# manually set start/end times for survey stops (exclude motion and transient noise)
# discharge average 2980
t1_list1 = ['21:02:09', '21:11:15', '21:22:13', '21:31:01', '21:40:00', '21:43:31', '21:49:29', 
            '21:52:58', '21:58:26', '22:02:23', '22:08:19', '22:11:57', '22:22:27', '22:25:31', 
            '22:34:01', '22:36:32.2', '22:48:55', '22:58:16', '23:04:22', '23:09:12', '23:21:49', '23:34:51', 
            '23:47:35']
t2_list1 = ['21:05:31', '21:12:54', '21:23:10', '21:34:30', '21:41:27', '21:45:36', '21:51:45', 
            '21:54:57', '22:00:01', '22:03:33', '22:09:11', '22:12:55', '22:23:25', '22:26:07', 
            '22:34:55', '22:36:40', '22:50:22', '23:00:09', '23:04:32', '23:11:01', '23:25:43', '23:35:29', 
            '23:49:07']
t1_list1 = np.array([obspy.UTCDateTime(f'2024-08-14 {t}') for t in t1_list1])
t2_list1 = np.array([obspy.UTCDateTime(f'2024-08-14 {t}') for t in t2_list1])

# calculate lat, lon, and RMS amplitude at each survey stop
lat_list = np.array([])
lon_list = np.array([])
rms_list = np.array([])
for i in range(len(t1_list1)):
    w = np.where((t >= t1_list1[i]) & (t <= t2_list1[i]))[0]
    lat_list = np.append(lat_list, np.mean(df_gps.lat.iloc[w]))
    lon_list = np.append(lon_list, np.mean(df_gps.lon.iloc[w]))
    rms_list = np.append(rms_list, tr.slice(t1_list1[i], t2_list1[i]).std() * 3.5012e-3)
    
#%% Calculate lats and lons for Aug 15 infrasound survey stops
df_gps = gemlog.read_gps('other_data/NiagaraFalls_GPS', SN = '266')
t = np.array([obspy.UTCDateTime(t) for t in df_gps.t])

tr = obspy.read('mseed/NiagaraFalls/2024-08-15T14_33_21..266..HDF.mseed')[0]
tr.filter('highpass', freq = 0.5)

# manually enter start/stop times at each location (excluding motion and transient noise)
# discharge average 2901
t1_list2 = ['14:39:49', '14:48:30', '15:08:28', '15:24:54', '16:19:30', '16:53:20', '17:20:00']
t2_list2 = ['14:40:13', '15:00:30', '15:09:50', '15:26:17', '16:26:30', '16:56:04', '17:21:40']
t1_list2 = np.array([obspy.UTCDateTime(f'2024-08-15 {t}') for t in t1_list2])
t2_list2 = np.array([obspy.UTCDateTime(f'2024-08-15 {t}') for t in t2_list2])

# Calculate lat, lon, and RMS amplitude for each stop
for i in range(len(t1_list2)):
    w = np.where((t >= t1_list2[i]) & (t <= t2_list2[i]))[0]
    lat_list = np.append(lat_list, np.mean(df_gps.lat.iloc[w]))
    lon_list = np.append(lon_list, np.mean(df_gps.lon.iloc[w]))
    rms_list = np.append(rms_list, tr.slice(t1_list2[i], t2_list2[i]).std() * 3.5012e-3)
    
# GPS didn't record at some sites; add these coordinates manually from Gaia waypoints.
lat_list[-2] = 43.07527; lon_list[-2] = -79.07976
lat_list[-4] = 43.07901; lon_list[-4] = -79.07840
lat_list[-5] = 43.08210; lon_list[-5] = -79.07790
lat_list[-1] = 43.08501; lon_list[-1] = -79.08226
#%% Plot the Aug 14-15 infrasound survey locations with dB difference as color
discharge = np.concatenate([np.repeat(2980.0, len(t1_list1)), np.repeat(2901.0, len(t1_list2))])
power_infrasound = 1e-6 * 1000 * 9.8 * discharge * df.height_m[5]
dB_diff = 20*np.log10(rms_list/20e-6) - model_NF_power(lon_list, lat_list, power_infrasound)
norm = TwoSlopeNorm(vmin=-np.max(np.abs(dB_diff)), vcenter=0, vmax=np.max(np.abs(dB_diff)))
sc = ax.scatter(lon_list, lat_list, c = dB_diff, norm = norm, cmap = 'bwr', s = 80, edgecolor = 'k')
cbar = plt.colorbar(sc, ax=ax, shrink = 0.333)
cbar.set_label('Observed - Modeled (dB)')

#%% save the figure
fig.tight_layout()
fig.savefig('figures/NiagaraFallsMap.png')
