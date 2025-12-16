import numpy as np
import time
from collections import deque
from scipy.signal import butter, lfilter, lfilter_zi
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load trained LDA model
lda = joblib.load('lda_gesture_model.pkl')

# Bandpass Filter Setup
fs = 1000.0
lowcut, highcut, order = 20.0, 450.0, 4
nyquist = 0.5 * fs
low, high = lowcut / nyquist, highcut / nyquist
b, a = butter(order, [low, high], btype='band')

# Buffer for filtered EMG data
window_size = 10
filtered_buffer = deque(maxlen=window_size)

def compute_zcr(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

def generate_values(min_val=150, max_val=1024, step_count=25, hold_count=10, delay=0.4):
    """
    Generates a sequence of values: increasing, holding at max, decreasing, holding at min.
    """
    increasing_values = np.linspace(min_val, max_val, step_count, dtype=int)
    decreasing_values = np.linspace(max_val, min_val, step_count, dtype=int)[::-1]
    while True:
        for val in increasing_values:
            yield val
        for _ in range(hold_count):
            yield max_val
        for val in decreasing_values:
            yield val
        for _ in range(hold_count):
            yield min_val

def generate_and_predict_gestures():
    zi = lfilter_zi(b, a) * 0  # Initial filter state
    
    for raw_emg in generate_values():
        filtered_emg, zi = lfilter(b, a, [raw_emg], zi=zi)
        filtered_emg = filtered_emg[0]
        filtered_buffer.append(filtered_emg)
        
        rms = np.sqrt(np.mean(np.square(filtered_buffer))) if filtered_buffer else 0
        sav = np.mean(list(filtered_buffer)[-2:]) if len(filtered_buffer) >= 2 else np.mean(filtered_buffer) if filtered_buffer else 0
        std = np.std(filtered_buffer) if filtered_buffer else 0
        zcr = compute_zcr(filtered_buffer) if filtered_buffer else 0
        
        features = np.array([[raw_emg, filtered_emg, rms, sav, std, zcr]])
        gesture = lda.predict(features)[0]
        
        print(f"Raw EMG: {raw_emg:.2f} | Filtered EMG: {filtered_emg:.2f} | Gesture: {gesture}")
        time.sleep(0.4)

if __name__ == "__main__":
    generate_and_predict_gestures()
