import numpy as np

def calc_mean_freq(f, s):
    return np.sum(f*s)/np.sum(s)
def calc_geo_mean_freq(f, s):
    return np.exp(np.sum(np.log(f[1:])*s[1:])/np.sum(s[1:]))
def calc_med_freq(f, s):
    try:
        return f[np.where(np.cumsum(s) > (0.5*np.sum(s)))[0][0]]
    except:
        return np.nan

temperature = 288 # Kelvin
pressure = 100000 # Pa
R = 287.05
sound_speed = np.sqrt(1.4 * R * temperature)
density = pressure / (R * temperature)
impedance = sound_speed * density
