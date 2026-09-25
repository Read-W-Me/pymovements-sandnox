# Read W/Me: Movement Classifier Validation

## Overview
This repository contains the `pymovements` gaze classification pipeline for the **Read W/Me** project. The core extraction logic relies on a Dispersion-Threshold (I-DT) algorithm to group continuous raw gaze coordinates into discrete cognitive macro-events (fixations, saccades, and regressions).

## Pipeline Architecture & HMM Integration
The Read W/Me architecture utilizes a multi-layered approach to signal processing. The layer preceding this classification pipeline employs Hidden Markov Models (HMM) to actively filter out hardware jitter, tracker loss, blinks, and high-frequency noise. 

Because the HMM layer delivers a pre-stabilized signal, utilizing a pure I-DT algorithm is the optimal design choice for this stage. It avoids the computational overhead of Velocity-Threshold (I-VT) parsing while effectively clustering the cleaned spatial data into actionable reading metrics.

## ETDD70 Validation Findings
The pipeline was validated against the academic ground truth of the ETDD70 dataset (250Hz hardware, pediatric readers). 

By establishing a dynamic hardware frequency detector and calibrating the I-DT parameters to a **100ms minimum duration** and a **2.2-degree dispersion threshold**, the custom pipeline achieved parity with commercial proprietary algorithms:
*   **Fixation Count:** 95% accuracy (229 extracted vs. 218 ground truth)
*   **Mean Fixation Duration:** 98% accuracy (610ms vs. 622ms)
*   **Regression Count:** 100% accuracy (15 extracted vs. 15 ground truth)

*Note on Saccade Durations:* Pure I-DT expands spatial clusters until they touch, which artificially compresses the recorded saccade flight time (~11ms). Because Read W/Me prioritizes cognitive macro-metrics (dwell time, regression rates, fixation counts) over micro-kinematics, this compression is an acceptable artifact.

## Future Calibration Requirements
The current `dispersion_threshold=2.2` and `minimum_duration=100ms` parameters are optimized for high-speed (250Hz) clinical trackers and the natural gaze drift of pediatric subjects reading difficult text. 

**Further calibration is mandatory** for the final Read W/Me system. The I-DT parameters must be re-tuned to match our specific production hardware capabilities, frame rate, spatial resolution, and target demographic baseline.
