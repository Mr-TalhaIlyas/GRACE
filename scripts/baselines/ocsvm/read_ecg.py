import pyedflib
import numpy as np

# Path to your EDF file
edf_file = 'path_to_your_file.edf'

# Read the EDF file
f = pyedflib.EdfReader(edf_file)

# Get the list of signal labels
signal_labels = f.getSignalLabels()

# Identify ECG channels (assuming labels are 'ECG+' and 'ECG-')
ecg_channel_indices = [i for i, label in enumerate(signal_labels) if 'ECG' in label]

# Extract ECG signals
ecg_signals = []
for idx in ecg_channel_indices:
    signal = f.readSignal(idx)
    ecg_signals.append(signal)

# Since it's a single-lead ECG, we can subtract ECG- from ECG+ if needed
if len(ecg_signals) == 2:
    ecg_signal = ecg_signals[0] - ecg_signals[1]
else:
    ecg_signal = ecg_signals[0]

# Get the sampling frequency
fs = f.getSampleFrequency(ecg_channel_indices[0])

# Close the EDF file
f._close()
del f
#%%
from scipy.signal import butter, filtfilt

def highpass_filter(signal, fs, cutoff=0.5, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Apply high-pass filter to remove baseline wander
ecg_filtered = highpass_filter(ecg_signal, fs, cutoff=0.5)

def notch_filter(signal, fs, freq=50.0, quality_factor=30.0):
    nyquist = 0.5 * fs
    norm_freq = freq / nyquist
    b, a = butter(2, [norm_freq - 0.5 / nyquist, norm_freq + 0.5 / nyquist], btype='bandstop')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Apply notch filter to remove powerline interference
ecg_filtered = notch_filter(ecg_filtered, fs, freq=50.0)

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Apply bandpass filter
ecg_filtered = bandpass_filter(ecg_filtered, fs, lowcut=0.5, highcut=40.0)

from biosppy.signals import ecg

# R-peak detection
out = ecg.ecg(signal=ecg_filtered, sampling_rate=fs, show=False)
rpeaks = out['rpeaks']

import neurokit2 as nk

# R-peak detection
rpeaks, info = nk.ecg_peaks(ecg_filtered, sampling_rate=fs)
rpeaks = rpeaks['ECG_R_Peaks']
