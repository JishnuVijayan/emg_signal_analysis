# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:15:57 2024

@author: jishn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_utilities as su
import emg as e

# Load EMG data
column_names = ['time', 'value']
emg_values = pd.read_csv('combined_rawdata.csv', sep=',', names=column_names)
emg_values = emg_values[0:600]
emg = emg_values['value']

time = emg_values['time'] / 1000

# Initialize the EMG filter
sample_frequency = 100  # Adjust as necessary
range_ = 0.1  # Adjust as necessary
min_EMG_frequency = 25  # Adjust as necessary
max_EMG_frequency = 600  # Adjust as necessary
reference_available = False  # Adjust as necessary

# Instantiate the EMG_filter class
filter = e.EMG_filter(sample_frequency=sample_frequency, range_=range_, min_EMG_frequency=min_EMG_frequency, max_EMG_frequency=max_EMG_frequency, reference_available=reference_available)

# Apply the filter to each EMG value
filtered_emg_values = []
for value in emg:
    filtered_value = filter.filter(value)
    filtered_emg_values.append(filtered_value)

# Plot only the filtered EMG signal
plt.figure(figsize=(10, 6))
plt.plot(time, filtered_emg_values, label='Filtered EMG', color='blue')  # Adjust color if needed
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered EMG Signal')
plt.legend()
plt.grid(True)
plt.show()

# Convert filtered EMG signal to DataFrame and save to CSV without labels
filtered_emg_df = pd.DataFrame({'time': time, 'filtered_emg': filtered_emg_values})

# Save filtered data to CSV without labels
output_file_path = 'For paper.csv'
filtered_emg_df.to_csv(output_file_path, index=False)
print("Filtered EMG data saved to:", output_file_path)