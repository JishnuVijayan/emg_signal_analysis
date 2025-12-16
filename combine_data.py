# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:18:17 2024

@author: jishn
"""

import pandas as pd
import os

# List of CSV file names
csv_files = [
    "01Blf.csv",
    "01BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "02Blf.csv",
    "02BRf.csv",
    "combined_data.csv"
]

# Initialize an empty list to hold the DataFrames
dfs = []

# Read each CSV file into a DataFrame and append it to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Drop the 'user' column if it exists
if 'user' in combined_df.columns:
    combined_df = combined_df.drop('user', axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("New Data for Training.csv", index=False)
