# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:30:57 2024

@author: jishn
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_utilities as su
import emg as e

# Load EMG data
column_names = ['time', 'value']
emg_values = pd.read_csv('combined_raw_down_data.csv', sep=',', names=column_names)
emg = emg_values['value']

time = emg_values['time'] / 1000

# Initialize the EMG filter
sample_frequency = 100 # Adjust as necessary
range_ = 0.1 # Adjust as necessary
min_EMG_frequency = 25 # Adjust as necessary
max_EMG_frequency = 600 # Adjust as necessary
reference_available = False # Adjust as necessary

# Instantiate the EMG_filter class
filter = e.EMG_filter(sample_frequency=sample_frequency, range_=range_, min_EMG_frequency=min_EMG_frequency, max_EMG_frequency=max_EMG_frequency, reference_available=reference_available)

# Apply the filter to each EMG value
filtered_emg_values = []
for value in emg:
    filtered_value = filter.filter(value)
    filtered_emg_values.append(filtered_value)

# Plot filtered EMG signal
plt.figure(figsize=(10, 6))
plt.plot(time, emg, label='Original EMG')
plt.plot(time, filtered_emg_values, label='Filtered EMG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered EMG Signal')
plt.legend()
plt.grid(True)
plt.show()

# Convert EMG signal to DataFrame
filtered_emg_df = pd.DataFrame({'time': time, 'filtered_emg': filtered_emg_values})

# Set the threshold based on the mean and standard deviation
mean_value = np.mean(filtered_emg_values)
std_value = np.std(filtered_emg_values)
threshold = std_value 

# Classify the hand's state based on the threshold
filtered_emg_df['gesture'] = 'Hand is at Rest'
filtered_emg_df.loc[filtered_emg_df['filtered_emg'] >= threshold, 'gesture'] = 'Hand is Moving DOWN'

# Add rolling window features
window_size = 12
rolling = filtered_emg_df['filtered_emg'].rolling(window=window_size)
filtered_emg_df['mean'] = rolling.mean()
filtered_emg_df['std'] = rolling.std()
filtered_emg_df['min'] = rolling.min()
filtered_emg_df['max'] = rolling.max()
filtered_emg_df['median'] = rolling.median()
# Drop the rows with NaN values caused by the rolling window
filtered_emg_df = filtered_emg_df.dropna()

# Save classified data to CSV
output_file_path = 'combined_filtered_down_data_new.csv'
filtered_emg_df.to_csv(output_file_path, index=False)
print("Filtered EMG data with added features saved to:", output_file_path)
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.fftpack
import matplotlib.pyplot as plt
import signal_utilities as su
import emg as e

# Function to calculate zero-crossing rate for a window of values
def zero_crossing_rate(window):
    return ((window[:-1] * window[1:]) < 0).sum()

# Function to calculate slope sign changes for a window of values
def slope_sign_changes(window):
    return ((np.diff(window)[:-1] * np.diff(window)[1:]) < 0).sum()

# Function to calculate waveform length for a window of values
def waveform_length(window):
    return np.sum(np.abs(np.diff(window)))

# Function to calculate spectral entropy for a window of values
def spectral_entropy(window):
    fft_values = np.abs(scipy.fftpack.fft(window))
    fft_values = fft_values / np.sum(fft_values)
    return stats.entropy(fft_values)

# Load EMG data
column_names = ['time', 'value']
emg_values = pd.read_csv('combined_raw_right_data.csv', sep=',', names=column_names)
emg = emg_values['value']

time = emg_values['time'] / 1000

sample_frequency = 100 # Adjust as necessary
range_ = 0.1 # Adjust as necessary
min_EMG_frequency = 25 # Adjust as necessary
max_EMG_frequency = 600 # Adjust as necessary
reference_available = False # Adjust as necessary

filter = e.EMG_filter(sample_frequency=sample_frequency, range_=range_, min_EMG_frequency=min_EMG_frequency, max_EMG_frequency=max_EMG_frequency, reference_available=reference_available)

# Apply the filter to each EMG value
filtered_emg_values = []
for value in emg:
    filtered_value = filter.filter(value)
    filtered_emg_values.append(filtered_value)

# Convert EMG signal to DataFrame
filtered_emg_df = pd.DataFrame({'time': time, 'filtered_emg': filtered_emg_values})

# Set the threshold based on the mean and standard deviation
mean_value = np.mean(filtered_emg_values)
std_value = np.std(filtered_emg_values)
threshold = std_value 

# Classify the hand's state based on the threshold
filtered_emg_df['gesture'] = 'Hand is at Rest'
filtered_emg_df.loc[filtered_emg_df['filtered_emg'] >= threshold, 'gesture'] = 'Hand is Moving RIGHT'

# Add rolling window features
window_size = 12
rolling = filtered_emg_df['filtered_emg'].rolling(window=window_size)
filtered_emg_df['mean'] = rolling.mean()
filtered_emg_df['std'] = rolling.std()
filtered_emg_df['min'] = rolling.min()
filtered_emg_df['max'] = rolling.max()
filtered_emg_df['median'] = rolling.median()

# Add new features
filtered_emg_df['zcr'] = filtered_emg_df['filtered_emg'].rolling(window=window_size).apply(zero_crossing_rate)
filtered_emg_df['ssc'] = filtered_emg_df['filtered_emg'].rolling(window=window_size).apply(slope_sign_changes)
filtered_emg_df['wl'] = filtered_emg_df['filtered_emg'].rolling(window=window_size).apply(waveform_length)

# Print DataFrame information
print(filtered_emg_df.head())

# Add new features
filtered_emg_df['se'] = filtered_emg_df['filtered_emg'].rolling(window=window_size).apply(lambda x: spectral_entropy(np.array(x)))

# Drop the rows with NaN values caused by the rolling window
filtered_emg_df = filtered_emg_df.dropna()

# Save classified data to CSV
output_file_path = 'combined_filtered_right_data_new_with_more_column.csv'
filtered_emg_df.to_csv(output_file_path, index=False)
print("Filtered EMG data with added features saved to:", output_file_path)
