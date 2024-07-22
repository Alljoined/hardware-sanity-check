import mne
import argparse
import numpy as np
from preprocessing_utils import mvnn, save_prepr

mne.set_log_level('WARNING')

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_ses', default=1, type=int)
parser.add_argument('--sfreq', default=512, type=int)
parser.add_argument('--mvnn_dim', default='epochs', type=str)
parser.add_argument('--project_dir', default='/srv/eeg_reconstruction/mit/stimulus-emotiv', type=str)
parser.add_argument('--lo_freq', default=0.1, type=float)
parser.add_argument('--hi_freq', default=100, type=float)
args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Load the cleaned data
# =============================================================================
train_fif_path = '/srv/eeg_reconstruction/mit/stimulus-emotiv/fif/subj01_session1_eeg_cleaned_train-epo.fif'
test_fif_path = '/srv/eeg_reconstruction/mit/stimulus-emotiv/fif/subj01_session1_eeg_cleaned_test-epo.fif'

epochs_train = mne.read_epochs(train_fif_path, preload=True)
epochs_test = mne.read_epochs(test_fif_path, preload=True)

# Extract data and necessary information
epoched_train = [epochs_train.get_data()]
epoched_test = [epochs_test.get_data()]
ch_names = epochs_train.ch_names

# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
# MVNN is applied independently to the data of each session.
whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)

# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
save_prepr(args, whitened_test, whitened_train, epochs_train.events, ch_names, seed)
