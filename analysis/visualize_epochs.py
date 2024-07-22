import numpy as np
import mne
import matplotlib.pyplot as plt

# Load the preprocessed data
whitened_test = np.load('/srv/eeg_reconstruction/mit/stimulus-emotiv/whitened_test.npy')
whitened_train = np.load('/srv/eeg_reconstruction/mit/stimulus-emotiv/whitened_train.npy')

# Example: Visualize the first condition from the test data
# Create an info object to store channel names and types
n_channels = 32
sfreq = 512  # Sample frequency
ch_names = ['EEG %02d' % i for i in range(1, n_channels + 1)]
ch_types = ['eeg'] * n_channels

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Extract the first epoch data (assuming shape is [conditions, repetitions, channels, time points])
# Reshape to (n_epochs, n_channels, n_samples)
epochs_data = whitened_test[0, :, :, :].reshape(-1, n_channels, whitened_test.shape[-1])

# Create MNE Epochs object
epochs = mne.EpochsArray(epochs_data, info)

# Plot the epochs with adjusted scaling
fig = epochs.plot(n_epochs=10, n_channels=n_channels, scalings=dict(eeg=20e-6))  # Adjust the scaling factor here
fig.savefig('/srv/eeg_reconstruction/mit/stimulus-emotiv/epochs_plot_rescaled.png')
