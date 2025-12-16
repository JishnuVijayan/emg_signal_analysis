# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:10:54 2024

@author: jishn
"""

import pandas as pd
import numpy as np

df = pd.read_csv("combined_all_data_new_with_more_columns.csv")
print(df.head())
df_shape = df.shape
print(df_shape)

# Suppose 'gesture' column contains different gestures
gestures = df['gesture'].unique()
num_gestures = len(gestures)

# Create a dictionary to map each gesture to a unique index
gesture_to_index = {gesture: index for index, gesture in enumerate(gestures)}

# Add a new dimension for gestures and reshape the data
num_samples = df.shape[0]
num_features = df.shape[1] - 1  # Exclude 'gesture' column
data_3d = np.zeros((num_samples, num_features, num_gestures))

# Iterate through each sample and populate the 3-D array
for i, (_, row) in enumerate(df.iterrows()):
    gesture_index = gesture_to_index[row['gesture']]
    data_3d[i, :, gesture_index] = row.drop('gesture')
print(data_3d)


