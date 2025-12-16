# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:26:55 2024

@author: jishn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_utilities as su
import emg as e

# Load EMG data from updated_jishnu.txt
column_names = ['time', 'value']
emg_values = pd.read_csv('new data salman.csv', sep=',', names=column_names)

# Convert 'time' column to numerical values (integers or floats)
emg_values['time'] = pd.to_numeric(emg_values['time'], errors='coerce')

# Drop any rows with NaN values that resulted from conversion
emg_values.dropna(inplace=True)

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

mean_value = np.mean(filtered_emg_values)
std_value = np.std(filtered_emg_values)
print(mean_value)
print(std_value)

# Set the threshold based on the mean and standard deviation
threshold = std_value 

# Classify the hand's state based on the threshold
filtered_emg_df['gesture'] = 'Hand is at Rest'
filtered_emg_df.loc[filtered_emg_df['filtered_emg'] >= threshold, 'gesture'] = 'Hand is Moving RIGHT'

# Save classified data to CSV
output_file_path = 'updated salman.csv'
filtered_emg_df.to_csv(output_file_path, index=False)
print("Filtered EMG data saved to:", output_file_path)
