#%%
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from biosppy.signals import ecg
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
from pyhrv.tools import plot_ecg
import pyhrv.nonlinear as nl
from scipy.stats import skew
from scipy.stats import kurtosis as get_kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path
import glob
from fmutils import fmutils as fmu

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def katz_fd(x, axis=-1):
    """Katz Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which the FD is calculated. Default is -1 (last).

    Returns
    -------
    kfd : float
        Katz fractal dimension.

    Notes
 
    """
    x = np.asarray(x)
    dists = np.abs(np.diff(x, axis=axis))
    ll = dists.sum(axis=axis)
    ln = np.log10(ll / dists.mean(axis=axis))
    aux_d = x - np.take(x, indices=[0], axis=axis)
    d = np.max(np.abs(aux_d), axis=axis)
    kfd = np.squeeze(ln / (ln + np.log10(d / ll)))
    if not kfd.ndim:
        kfd = kfd.item()
    return kfd
#%%

def extract_ecg_features(ecg_signal_path, labels_path):

    assert Path(ecg_signal_path).stem == Path(labels_path).stem.replace('_ecg_lbl', '')
    
    ecg_signal = np.load(ecg_signal_path)
    labels = np.load(labels_path)

    fname = Path(ecg_signal_path).stem
    fs = 250 

    filtered_indices = np.where(labels != 5)[0]

    labels = labels[filtered_indices]
    ecg_signal = ecg_signal[filtered_indices]
    labels = np.clip(labels, 0, 1)

    # Baseline correction and filtering
    ecg_detrended = detrend(ecg_signal)
    ecg_filtered = bandpass_filter(ecg_detrended, lowcut=1, highcut=50, fs=fs, order=5)

    # R-peak detection
    out = ecg.ecg(signal=ecg_filtered, sampling_rate=fs, show=False)
    rpeaks = out['rpeaks']

    # RR intervals and times
    rr_intervals = np.diff(rpeaks) / fs
    rr_times = rpeaks[1:] / fs

    # Artifact removal
    L = 30
    tau = 0.5
    rr_series = pd.Series(rr_intervals)
    rr_mavg = rr_series.rolling(window=L, min_periods=1, center=True).mean()
    rr_clean = rr_intervals.copy()
    for i in range(len(rr_intervals)):
        if np.abs(rr_intervals[i] - rr_mavg[i]) > tau * np.abs(rr_mavg[i]):
            rr_clean[i] = rr_mavg[i]

    # 3. Segment NN series
    window_size_sec = 2 * 60
    mean_rr = np.mean(rr_clean)
    window_size_samples = int(window_size_sec / mean_rr)
    step_size = 1
    segments = []
    segment_times = []

    for start in range(0, len(rr_clean) - window_size_samples + 1, step_size):
        end = start + window_size_samples
        rr_segment = rr_clean[start:end]
        time_segment = rr_times[start:end]
        segments.append(rr_segment)
        segment_times.append(time_segment[-1])

    # 4. Extract HRV features
    features_list = []

    for rr_segment in tqdm(segments):
        nn_intervals = rr_segment * 1000  # Convert to milliseconds

        # Time-domain features
        time_domain_features = td.time_domain(nn_intervals, show=False, plot=False)
        mean_nni = time_domain_features['nni_mean']
        sdnn = time_domain_features['sdnn']
        rmssd = time_domain_features['rmssd']
        sdsd = time_domain_features['sdsd']
        nn50 = time_domain_features['nn50']
        pnn50 = time_domain_features['pnn50']
        skewness = skew(nn_intervals)
        kurtosis = get_kurtosis(nn_intervals)

        # Frequency-domain features
        freq_domain_features = fd.welch_psd(nn_intervals, show=False, mode='dev')[0]
        lf = freq_domain_features['fft_abs'][1]
        hf = freq_domain_features['fft_abs'][2]
        lf_hf_ratio = freq_domain_features['fft_ratio']
        lf_peak = freq_domain_features['fft_peak'][1]
        hf_peak = freq_domain_features['fft_peak'][2]

        # Non-linear features
        sampen = nl.sample_entropy(nn_intervals)['sampen']
        poincare_results = nl.poincare(nn_intervals, show=False, mode='dev')
        sd1 = poincare_results['sd1']
        sd2 = poincare_results['sd2']
        sd1_sd2_ratio = sd1 / sd2 if sd2 != 0 else 0
        ellipse_area = np.pi * sd1 * sd2
        kfd = katz_fd(nn_intervals)

        # Collect features
        features = [
            mean_nni,
            sdnn,
            skewness,
            kurtosis,
            nn50,
            rmssd,
            sdsd,
            sampen,
            sd1,
            sd2,
            sd1_sd2_ratio,
            ellipse_area,
            kfd,
            lf,
            hf,
            lf_hf_ratio,
            lf_peak,
            hf_peak,
        ]
        features_list.append(features)

    # 5. Reduce dimensionality with PCA
    X = np.array(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'Extracted RAW features for {fname} => {X.shape}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)


    segment_labels = []

    for time_segment in segment_times:
        index = int(time_segment * fs)
        segment_label = labels[index]
        segment_labels.append(segment_label)

    segment_labels = np.array(segment_labels)

    if 'b_' in fname:
        print('Assigning labels for PNES')
        segment_labels = segment_labels * 2
    
    print(f'Extracted PCA features => {X_pca.shape}')
    print(f'Labels => {segment_labels.shape}')

    return X_scaled, X_pca, segment_labels


# ecg_signal = np.load("/home/user01/Data/mme/dataset/ecg_arr/a_patient_1.npy")  # Raw ECG data
# labels = np.load("/home/user01/Data/mme/dataset/labels/a_patient_1_ecg_lbl.npy")          # Labels corresponding to ECG data

# ecg_signal_path = "/home/user01/Data/mme/dataset/ecg_arr/a_patient_1.npy"
# labels_path = "/home/user01/Data/mme/dataset/labels/a_patient_1_ecg_lbl.npy"

op_dir = '/home/user01/Data/mme/dataset/baseline_feats/ocsvm/'

ecg_dir = '/home/user01/Data/mme/dataset/ecg_arr/'
lbl_dir = '/home/user01/Data/mme/dataset/labels/'

ecg_files = glob.glob(ecg_dir + '*.npy')
lbl_files = glob.glob(lbl_dir + '*_ecg_lbl.npy')

ecg_files = sorted(ecg_files, key=fmu.numericalSort)
lbl_files = sorted(lbl_files, key=fmu.numericalSort)

all_features = []
all_pca_features = []
all_labels = []
for i in range(len(ecg_files)):
    X_scaled, X_pca, segment_labels = extract_ecg_features(ecg_files[i], lbl_files[i])

    all_features.append(X_scaled)
    all_pca_features.append(X_pca)
    all_labels.append(segment_labels)

all_features = np.concatenate(all_features, axis=0)
all_pca_features = np.concatenate(all_pca_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f'All Features => {all_features.shape}')
print(f'All PCA Features => {all_pca_features.shape}')
print(f'All Labels => {all_labels.shape}')

np.save(op_dir + 'all_features.npy', all_features)
np.save(op_dir + 'all_pca_features.npy', all_pca_features)
np.save(op_dir + 'all_labels.npy', all_labels)

# plot_ecg(ecg_signal, interval=[300,310])

#%%