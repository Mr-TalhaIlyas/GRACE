#%%
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from biosppy.signals import ecg
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

exp_typ = 'gvp' # gvp or all

X_pca = np.load('/home/user01/Data/mme/dataset/baseline_feats/ocsvm/all_pca_features.npy')
segment_labels = np.load('/home/user01/Data/mme/dataset/baseline_feats/ocsvm/all_labels.npy')
# segment_labels = np.clip(segment_labels, 0, 1)

if exp_typ == 'gvp': # gtcs vs pnes
    # filter incdices based on the labels
    idx = np.where(segment_labels != 0) # removed basline
    X_pca = X_pca[idx]
    segment_labels = segment_labels[idx] - 1 # to make it 0 based
    # 0 is gtcs 1 is pnes so gtcs is normal and pnes is abnormal
else:  # preictal vs ictal
    X_pca = X_pca
    segment_labels = np.clip(segment_labels, 0, 1)



#%%
# split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, segment_labels, test_size=0.2, random_state=42)

# Train on only normal data in the training set (label == 0)
X_train_normal = X_train[y_train == 0]

# Initialize and train the One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(X_train_normal)

# Make predictions on the test set
y_pred = ocsvm.predict(X_test)
anomaly_scores = -ocsvm.decision_function(X_test)

# Convert predictions to labels: 1 for anomalies (seizures), 0 for normal
y_pred_labels = (y_pred == -1).astype(int)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=['Normal', 'Seizure']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))

x = classification_report(y_test, y_pred_labels, target_names=['Normal', 'Seizure'], output_dict=True)
#%%
