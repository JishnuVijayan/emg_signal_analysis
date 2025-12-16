# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# # Load the dataset
# file_path = 'combined_emg_data.csv'
# data = pd.read_csv(file_path)

# # Convert time to numeric type if it's not already
# data['time'] = pd.to_numeric(data['time'], errors='coerce')
# data['filtered_emg'] = pd.to_numeric(data['filtered_emg'], errors='coerce')

# # Plot the EMG signal over time
# plt.figure(figsize=(12, 6))

# # Plot the 'filtered_emg' over time, color the segments based on the gesture
# colors = {'Hand is at Rest': 'blue', 'Hand is Moving': 'red'}

# # Create the plot
# for i in range(len(data) - 1):
#     # Get the segment start and end
#     start_time = data.iloc[i]['time']
#     end_time = data.iloc[i + 1]['time']
#     gesture = data.iloc[i]['gesture']
    
#     # Plot a line segment for each time slice
#     plt.plot([start_time, end_time], [data.iloc[i]['filtered_emg'], data.iloc[i + 1]['filtered_emg']], 
#              color=colors[gesture], lw=2)

# # Add labels and title
# plt.xlabel('Time (seconds)')
# plt.ylabel('Filtered EMG')
# plt.title('Filtered EMG Signal Over Time')

# # Add a legend for gestures
# plt.legend(colors.keys(), title='Gestures', loc='upper right')

# # Display the plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'combined_emg_data.csv'
data = pd.read_csv(file_path)

# Convert time and filtered_emg to numeric
data['time'] = pd.to_numeric(data['time'], errors='coerce')
data['filtered_emg'] = pd.to_numeric(data['filtered_emg'], errors='coerce')

# Identify where time = 0.0 to determine when a new person's data starts
data['new_person'] = data['time'] == 0.0

# Initialize a list to keep track of person start indices
person_indices = [0]  # Start with the first entry
person_indices.extend(data[data['new_person']].index.tolist())  # Add indices where new_person == True

# Plotting each person's data in a separate graph
fig, axes = plt.subplots(len(person_indices) - 1, 1, figsize=(12, 6), sharex=True)

# Ensure we have a single axis for the case where there's only one person
if len(person_indices) - 1 == 1:
    axes = [axes]

for i in range(len(person_indices) - 1):
    start_idx = person_indices[i]
    end_idx = person_indices[i + 1] - 1  # Next person's start is the end of current person's data

    # Slice the data for the current person
    person_data = data.iloc[start_idx:end_idx+1]
    
    # Skip empty person data
    if person_data.empty:
        continue
    
    # Reset time for each person
    person_data['time'] = person_data['time'] - person_data['time'].iloc[0]

    # Get the gesture for color coding
    colors = {'Hand is at Rest': 'blue', 'Hand is Moving': 'red'}
    
    # Plot the person's data
    for j in range(len(person_data) - 1):
        start_time = person_data.iloc[j]['time']
        end_time = person_data.iloc[j + 1]['time']
        gesture = person_data.iloc[j]['gesture']
        
        axes[i].plot([start_time, end_time], 
                     [person_data.iloc[j]['filtered_emg'], person_data.iloc[j + 1]['filtered_emg']], 
                     color=colors[gesture], lw=2)

    # Label each subplot
    axes[i].set_title(f"Person {i+1} EMG Signal")
    axes[i].set_ylabel('Filtered EMG')
    axes[i].grid(True)

# Set common x-axis label
plt.xlabel('Time (seconds)')
plt.tight_layout()

# Display the plot
plt.show()
