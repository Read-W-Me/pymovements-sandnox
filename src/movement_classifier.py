import pymovements as pm
import pandas as pd
import numpy as np
import os

def extract_reading_parameters(events_pandas, raw_df, sampling_rate_hz, line_height_px=40, noise_threshold_px=15, aoi_bounds=None):
    if events_pandas.empty:
        return {}
        
    ms_per_frame = 1000.0 / sampling_rate_hz

    if 'centroid_x' not in events_pandas.columns:
        c_x, c_y = [], []
        for _, row in events_pandas.iterrows():
            mask = (raw_df['time_idx'] >= row['onset']) & (raw_df['time_idx'] <= row['offset'])
            c_x.append(raw_df.loc[mask, 'x'].mean())
            c_y.append(raw_df.loc[mask, 'y'].mean())
        events_pandas['centroid_x'] = c_x
        events_pandas['centroid_y'] = c_y

    fixations = events_pandas[events_pandas['name'] == 'fixation'].copy()
    saccades = events_pandas[events_pandas['name'] == 'saccade'].copy()
    
    fixation_count = len(fixations)
    fixation_dur_mean = fixations['duration'].mean() * ms_per_frame if fixation_count > 0 else 0
    fixation_dur_std = fixations['duration'].std() * ms_per_frame if fixation_count > 1 else 0

    if saccades.empty and fixation_count > 1:
        saccade_onsets = fixations['offset'].iloc[:-1].values
        saccade_offsets = fixations['onset'].iloc[1:].values
        saccade_durations = saccade_offsets - saccade_onsets
        
        valid_saccades = saccade_durations > 0
        saccade_count = int(valid_saccades.sum())
        saccade_dur_mean = float(saccade_durations[valid_saccades].mean()) * ms_per_frame if saccade_count > 0 else 0
    else:
        saccade_count = len(saccades)
        saccade_dur_mean = saccades['duration'].mean() * ms_per_frame if saccade_count > 0 else 0
        
    s_f_ratio = saccade_count / fixation_count if fixation_count > 0 else 0

    regression_count = 0
    if fixation_count > 1:
        x_diffs = fixations['centroid_x'].diff()
        y_diffs = fixations['centroid_y'].diff()
        
        same_line_regression = (x_diffs < -noise_threshold_px) & (y_diffs.abs() < line_height_px)
        previous_line_regression = (y_diffs < -line_height_px)
        regression_count = (same_line_regression | previous_line_regression).sum()

    aoi_metrics = {}
    if aoi_bounds and not fixations.empty:
        x_min, x_max, y_min, y_max = aoi_bounds
        in_aoi = (
            (fixations['centroid_x'] >= x_min) & (fixations['centroid_x'] <= x_max) &
            (fixations['centroid_y'] >= y_min) & (fixations['centroid_y'] <= y_max)
        )
        aoi_fixations = fixations[in_aoi]
        
        aoi_metrics['dwell_time'] = aoi_fixations['duration'].sum() * ms_per_frame
        aoi_metrics['first_fixation_latency'] = aoi_fixations['onset'].min() * ms_per_frame if not aoi_fixations.empty else None
        
        transitions = in_aoi.astype(int).diff()
        aoi_metrics['revisit_count'] = max(0, (transitions == 1).sum() - 1) if (transitions == 1).sum() > 0 else 0

    return {
        'fixation_count': fixation_count,
        'fixation_dur_mean': fixation_dur_mean,
        'fixation_dur_std': fixation_dur_std,
        'saccade_count': saccade_count,
        'saccade_dur_mean': saccade_dur_mean,
        's_f_ratio': s_f_ratio,
        'regression_count': int(regression_count),
        'aoi_metrics': aoi_metrics
    }

def process_real_gaze_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Dynamically calculate true sampling rate from microseconds
    time_diff_us = df['time'].diff().median()
    true_sampling_rate_hz = 1_000_000.0 / time_diff_us
    print(f"Detected True Hardware Sampling Rate: {true_sampling_rate_hz:.1f} Hz")

    # Map columns AND rename the original 'time' column so pymovements doesn't crash
    df.rename(columns={
        'gaze_x_left': 'x', 
        'gaze_y_left': 'y',
        'time': 'hardware_time' # Frees up the 'time' name for pymovements
    }, inplace=True)

    total_frames = len(df)
    valid_mask = df['x'].notna() & (df['x'] != 0) & df['y'].notna() & (df['y'] != 0)
    usable_frames = valid_mask.sum()
    usable_sample_pct = (usable_frames / total_frames) * 100
    tracking_loss_pct = 100 - usable_sample_pct

    df['x'] = df['x'].replace(0, np.nan).interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].replace(0, np.nan).interpolate(method='linear', limit_direction='both')

    # Keep a uniform frame index for pymovements
    df['time_idx'] = np.arange(len(df))

    experiment = pm.Experiment(
        screen_width_px=1920, screen_height_px=1080, 
        screen_width_cm=53.0, screen_height_cm=30.0, 
        distance_cm=60.0, sampling_rate=true_sampling_rate_hz
    )

    gaze = pm.gaze.from_pandas(
        df, experiment=experiment, time_column='time_idx', pixel_columns=['x', 'y']
    )

    gaze.pix2deg()
    gaze.pos2vel()

    # Convert our 100ms biological threshold to accurate hardware frames
    min_frames = int(100 / (1000 / true_sampling_rate_hz))
    gaze.detect('idt', dispersion_threshold=2.2, minimum_duration=min_frames)
    
    params = extract_reading_parameters(
        events_pandas=gaze.events.frame.to_pandas(), 
        raw_df=df,
        sampling_rate_hz=true_sampling_rate_hz,
        line_height_px=40,
        noise_threshold_px=15,
        aoi_bounds=(0, 1920, 0, 1080)
    )
    
    params['usable_sample_pct'] = usable_sample_pct
    params['tracking_loss_pct'] = tracking_loss_pct
    
    return params

if __name__ == "__main__":
    import os
    
    # Dynamically resolve path: goes up two levels from src/ (to Read_With_Me/) then down to Datasets/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '../../Datasets/ETDD70/raw_data'))
    
    filepath = None
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if 'Subject_1322_T5_Pseudo_Text_raw.csv' in files:
                filepath = os.path.join(root, 'Subject_1322_T5_Pseudo_Text_raw.csv')
                break
                
    if not filepath:
        print(f"File not found anywhere inside {base_dir}. Ensure the fetch script extracted it.")
    else:
        parameters = process_real_gaze_data(filepath)
        print("\n--- Extracted ETDD70 Reading Parameters ---")
        for key, value in parameters.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")