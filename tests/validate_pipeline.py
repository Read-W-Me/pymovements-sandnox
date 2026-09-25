import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import your pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.movement_classifier import process_real_gaze_data

def run_validation():
    # Dynamically resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '../../Datasets/ETDD70/raw_data'))
    
    raw_filepath = None
    metrics_filepath = None
    
    # Locate the raw and metrics files for Subject 1322 T5 Pseudo Text
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if 'Subject_1322_T5_Pseudo_Text_raw.csv' in files:
                raw_filepath = os.path.join(root, 'Subject_1322_T5_Pseudo_Text_raw.csv')
            if 'Subject_1322_T5_Pseudo_Text_metrics.csv' in files:
                metrics_filepath = os.path.join(root, 'Subject_1322_T5_Pseudo_Text_metrics.csv')
                    
    if not raw_filepath or not metrics_filepath:
        print(f"Could not find both the raw CSV and metrics CSV for Subject 1322 in {base_dir}.")
        print("Make sure the metrics files weren't deleted from the dataset folder.")
        return

    print("Running custom I-DT pipeline extraction...")
    # 1. Get our pipeline's output (Now dynamically calculates hardware sampling rate)
    our_params = process_real_gaze_data(raw_filepath)
    
    # 2. Get Ground Truth (GT) output
    gt_df = pd.read_csv(metrics_filepath, sep=None, engine='python')
    gt_df.columns = gt_df.columns.str.strip().str.lower()
    
    # The metrics file repeats trial data per AOI, so we take the first row for global trial metrics
    gt = gt_df.iloc[0]
    
    gt_params = {
        'fixation_count': gt.get('n_fix_trial', 0),
        'fixation_dur_mean': gt.get('mean_fix_dur_trial', 0),
        'saccade_count': gt.get('n_sacc_trial', 0),
        'saccade_dur_mean': gt.get('mean_sacc_dur_trial', 0),
        'regression_count': gt.get('n_regress_trial', 0)
    }

    # 3. Physiological Sanity Check
    print("\n" + "="*50)
    print(" PHYSIOLOGICAL BIOMETRIC CHECK ")
    print("="*50)
    
    s_f_ratio = our_params['s_f_ratio']
    regress_pct = (our_params['regression_count'] / our_params['fixation_count']) * 100 if our_params['fixation_count'] > 0 else 0
    
    checks = [
        ("Saccade Duration", our_params['saccade_dur_mean'], 15, 60, "ms"),
        ("Fixation Duration", our_params['fixation_dur_mean'], 150, 800, "ms (High end expected for dyslexia/pseudo-text)"),
        ("Saccade/Fixation Ratio", s_f_ratio, 0.8, 1.2, ""),
        ("Regression Rate", regress_pct, 5, 25, "%")
    ]
    
    for name, val, v_min, v_max, unit in checks:
        status = "✅ PASS" if v_min <= val <= v_max else "❌ FAIL"
        print(f"{name:<25}: {val:>6.2f} {unit:<45} | Bounds: [{v_min}-{v_max}] -> {status}")

    # 4. Engineering Comparison Matrix
    print("\n" + "="*50)
    print(" ENGINEERING CHECK: CUSTOM PIPELINE VS GROUND TRUTH ")
    print("="*50)
    
    metrics_to_compare = [
        ('Fixation Count', 'fixation_count'),
        ('Mean Fixation Dur (ms)', 'fixation_dur_mean'),
        ('Saccade Count', 'saccade_count'),
        ('Mean Saccade Dur (ms)', 'saccade_dur_mean'),
        ('Regression Count', 'regression_count')
    ]
    
    print(f"{'Metric':<25} | {'Custom Pipeline':<15} | {'Ground Truth (ETDD70)':<25} | {'Diff'}")
    print("-" * 80)
    
    plot_data_ours = []
    plot_data_gt = []
    labels = []
    
    for label, key in metrics_to_compare:
        our_val = our_params[key]
        gt_val = gt_params[key]
        diff = our_val - gt_val
        
        plot_data_ours.append(our_val)
        plot_data_gt.append(gt_val)
        labels.append(label)
        
        print(f"{label:<25} | {our_val:<15.2f} | {gt_val:<25.2f} | {diff:>+8.2f}")

    # 5. Generate Comparative Plots
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle('Pipeline Validation: Custom I-DT vs Dataset Ground Truth', fontsize=16)
    
    x = np.arange(1)  
    width = 0.35  

    for i, (label, our_v, gt_v) in enumerate(zip(labels, plot_data_ours, plot_data_gt)):
        ax = axes[i]
        bars1 = ax.bar(x - width/2, [our_v], width, label='Custom Pipeline', color='#2ca02c')
        bars2 = ax.bar(x + width/2, [gt_v], width, label='Ground Truth', color='#1f77b4')
        
        ax.set_title(label)
        ax.set_xticks([])
        
        # Add value text on top of bars
        ax.bar_label(bars1, fmt='%.1f', padding=3)
        ax.bar_label(bars2, fmt='%.1f', padding=3)
        
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'validation_comparison.png')
    plt.savefig(plot_path)
    print(f"\nValidation plots saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_validation()