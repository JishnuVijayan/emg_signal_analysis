import pandas as pd
import matplotlib.pyplot as plt

# Load the normalized dataset
df = pd.read_csv("normalized_data.csv")

# Take the first 100 rows for plotting
df_plot = df.head(100)

# Plot the filtered_emg signal over time
plt.figure(figsize=(12, 6))
plt.plot(df_plot['time'], df_plot['filtered_emg'], label='Filtered EMG', color='blue')

# Overlay the gesture column to highlight changes
gesture_changes = df_plot[df_plot['gesture'].diff() != 0]  # Find where the gesture changes
for _, row in gesture_changes.iterrows():
    plt.axvline(x=row['time'], color='red', linestyle='--', alpha=0.5, label='Gesture Change' if _ == gesture_changes.index[0] else "")

# Add labels and title
plt.title('Filtered EMG Signal Over Time', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Filtered EMG', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()