import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Load your data
x_test = pd.read_csv('x_test_new.csv')
plt.ion()
# Initialize the plot (it will be used throughout the loop)
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='g')

# Set labels and axis limits
ax.set_xlabel("Index")
ax.set_ylabel("Values")
ax.set_xlim(0, len(x_test) - 1)
ax.set_ylim(x_test.min().min(), x_test.max().max())  # Adjusting to data range

# Initialize data
x_data, y_data = [], []

# Update function for animation
def update(frame):
    global x_data, y_data
    
    # Append new data for the current frame
    x_data.append(frame)  # Use frame as the index
    y_data.append(x_test.iloc[frame].values[0])  # Only plot one value, assuming it's a single column for each row
    
    # Update the plot data
    line.set_data(x_data, y_data)
    return line,

# Create the animation
anim = FuncAnimation(fig, update, frames=len(x_test), interval=2000, blit=True)

# Display the animation
plt.show()
