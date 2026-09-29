import pandas as pd
import numpy as np
import os

def create_reading_mock_data():
    # 50 Hz sampling rate yields a constant 20ms integer interval
    sampling_rate = 50 
    interval_ms = 20
    
    def make_fixation(x, y, duration_ms):
        frames = int(duration_ms / interval_ms)
        x_arr = np.random.normal(x, 2, frames)
        y_arr = np.random.normal(y, 2, frames)
        return x_arr, y_arr
        
    def make_saccade(x_start, y_start, x_end, y_end, duration_ms):
        frames = max(1, int(duration_ms / interval_ms))
        x_arr = np.linspace(x_start, x_end, frames)
        y_arr = np.linspace(y_start, y_end, frames)
        return x_arr, y_arr

    xs, ys = [], []
    
    # 1. Forward reading (Line 1: Y=100)
    f_x, f_y = make_fixation(200, 100, 300)
    xs.append(f_x); ys.append(f_y)
    
    s_x, s_y = make_saccade(200, 100, 400, 100, 40)
    xs.append(s_x); ys.append(s_y)
    
    f_x, f_y = make_fixation(400, 100, 300)
    xs.append(f_x); ys.append(f_y)
    
    # 2. Same-line regression (Moves left: 400 -> 300)
    s_x, s_y = make_saccade(400, 100, 300, 100, 40)
    xs.append(s_x); ys.append(s_y)
    
    f_x, f_y = make_fixation(300, 100, 300)
    xs.append(f_x); ys.append(f_y)
    
    # 3. Return sweep to Line 2 (Moves left and down: Y=150)
    s_x, s_y = make_saccade(300, 100, 200, 150, 60)
    xs.append(s_x); ys.append(s_y)
    
    f_x, f_y = make_fixation(200, 150, 300)
    xs.append(f_x); ys.append(f_y)

    # 4. Previous-line regression (Moves up: Y=150 -> 100)
    s_x, s_y = make_saccade(200, 150, 250, 100, 60)
    xs.append(s_x); ys.append(s_y)
    
    f_x, f_y = make_fixation(250, 100, 300)
    xs.append(f_x); ys.append(f_y)
    
    # 5. Jump to AOI (Top right corner: bounds 1200-1920, 0-540)
    s_x, s_y = make_saccade(250, 100, 1500, 300, 60)
    xs.append(s_x); ys.append(s_y)
    
    f_x, f_y = make_fixation(1500, 300, 500)
    xs.append(f_x); ys.append(f_y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    n_samples = len(x)
    
    # Generate perfectly constant integer timestamps
    time = np.arange(0, n_samples) * interval_ms

    df = pd.DataFrame({'time': time, 'x': x, 'y': y})
    
    os.makedirs('data/mock_input', exist_ok=True)
    filepath = 'data/mock_input/basic_gaze.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Mock 50Hz reading data generated at: {filepath}")
    print(f"Total frames: {n_samples} | Total time: {time[-1]} ms")

if __name__ == "__main__":
    create_reading_mock_data()