# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:15:07 2024

@author: jishn
"""

import csv

# Function to calculate the mean of a list
def calculate_mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

# Read the CSV file
data = []
with open('combined_filtered_right_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

# Variables to store means and count of spikes
spike_means = []
spike_count = 0
spike_sum = 0

# Iterate through the data
for i in range(len(data)):
    if data[i][2] == "Hand is Moving RIGHT":
        # Start of a spike
        spike = [int(data[i][1])]
        j = i + 1
        consecutive_count = 1
        # Check consecutive rows for the same gesture and limit to 10 consecutive times
        while j < len(data) and data[j][2] == "Hand is Moving RIGHT" and consecutive_count < 10:
            spike.append(int(data[j][1]))
            j += 1
            consecutive_count += 1
        if consecutive_count == 10:
            # Calculate the mean of the spike
            spike_mean = calculate_mean(spike)
            spike_means.append(spike_mean)
            spike_count += 1
            spike_sum += spike_mean

# Calculate the overall mean of spike means
overall_mean = calculate_mean(spike_means)

print("Number of spikes:", spike_count)
print("Total mean of spikes:", spike_sum)
print("Overall mean of spike means:", overall_mean)
