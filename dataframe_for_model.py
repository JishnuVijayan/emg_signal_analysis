# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:40:44 2024

@author: jishn
"""
'''
import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 6):
    # Read the data from the file
    file_name = f"filtered_emg_data0{i}.csv"
    data = pd.read_csv(file_name)
    
    # Add a new column for the user
    data['user'] = i
    
    # Reorder the columns
    data = data[['time', 'user', 'filtered_emg', 'gesture']]
    
    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_data.csv", index=False)
'''

"""
import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 6):
    # Read the data from the file
    file_name = f"moredata0{i}.txt"
    data = pd.read_csv(file_name)
    
   
    
    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_rawdata.csv", index=False)
"""


import pandas as pd

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 6):
    # Read the data from the file
    file_name = f"moredata0{i}.txt"
    data = pd.read_csv(file_name, sep=',', header=None, names=['time', 'value'])
    
    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data])

# Save the combined data to a new file
combined_data.to_csv("combined_rawdata.csv", index=False)
