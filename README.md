# Wearable Strain Sensor Fusion & Validation Pipeline

Cross-validation framework for machine-knitted strain sensors integrated into smart leggings, using Xsens Awinda IMU data as ground truth for gait analysis.


## Overview

This pipeline calibrates knee-mounted strain sensors to biomechanical angles and validates their accuracy against IMU-derived kinematics using gyroscope integration. The approach enables low-cost wearable monitoring of knee flexion dynamics without relying solely on proprietary motion capture systems.

### Key Features
- **Strain-to-Angle Calibration**: Polynomial regression (0°–90° knee flexion range)
- **IMU Fusion**: Gyroscope integration for independent angle estimation
- **Cross-Validation**: RMSE, correlation, Bland-Altman analysis
- **Synthetic IMU Generation**: Test pipeline without real hardware constraints
- **Bilateral Support**: Simultaneous left/right knee monitoring

## Hardware

- **Strain Sensors**: Machine-knitted conductive fabric (110mm × 100mm), sewn into leggings at knee and shank locations
- **IMUs**: 4× Xsens Awinda (thigh × 2, shank lateral × 2)
- **Microcontroller**: Arduino Nano 33 IoT (BLE data streaming)
- **Sampling Rate**: 100 Hz

## Pipeline

### 1. Data Collection (`Visualize_both_knees_EXPORT.py`)

Real-time BLE streaming and visualization of bilateral strain sensor signals with CSV export.

**Protocol:**
- Static poses (0°, 30°, 60°, 90° knee flexion) for calibration
- Continuous gait walk (4–5 cycles) for validation
- Both knees simultaneously

**Output**: `strain_data_YYYYMMDD_HHMMSS.csv`

### 2. Data Merging (`merge_strain_csv.py`)

Asynchronous left/right readings are merged via time-based interpolation to common timeline.

```bash
python merge_strain_csv.py strain_data_YYYYMMDD_HHMMSS.csv
```

**Output**: `strain_data_YYYYMMDD_HHMMSS_merged.csv`

### 3. Calibration & Synthetic IMU (`generate_synthetic_imu.py`)

Calibrates strain sensor resistance (Ω) → knee flexion angle (°) using polynomial regression on static poses.

Generates synthetic 6-DOF IMU data (acceleration, angular velocity, magnetometry) consistent with strain-derived kinematics.

```bash
python generate_synthetic_imu.py
```

**Inputs:**
- Merged strain CSV
- Calibration angles: 0°, 30°, 60°, 90° (user-provided measurements)

**Outputs:**
- `synthetic_imu_gait.csv` — Accel, gyro, magnetometer data
- `strain_angles_calibrated.csv` — Full session angles
- `strain_to_imu_validation.png` — Calibration visualization

### 4. Fusion & Validation (`fusion_validation_pipeline.py`)

Integrates synthetic (or real) IMU gyroscope signals to estimate knee angle independently, then compares against strain-derived reference.

```bash
python fusion_validation_pipeline.py
```

**Metrics:**
- **RMSE**: Root mean squared error (degrees)
- **MAE**: Mean absolute error
- **Correlation**: Pearson r
- **Bland-Altman Plot**: Mean difference ± 1.96 SD limits of agreement
- **Error Distribution**: Histogram of residuals

**Outputs:**
- `strain_imu_fusion_validation.png` — Comparison plots, error analysis
- `fused_strain_imu_gait.csv` — Synchronized angles (strain vs IMU)

## Data Formats

### strain_data_merged.csv
```
timestamp_s,left_knee_ohms,right_knee_ohms
55.247,11.01,15.39
55.257,11.01,15.39
...
```

### synthetic_imu_gait.csv
```
timestamp_s,left_knee_angle_deg,left_shank_accel_x,left_shank_accel_y,...,left_shank_gyro_y,...
90.049,96.71,-0.386,9.743,...,-0.968,...
...
```

### fused_strain_imu_gait.csv
```
timestamp_s,left_knee_angle_deg,...,left_angle_imu_integrated,left_angle_strain,...
90.049,96.71,...,96.65,96.71,...
...
```

## Calibration Details

**Static Pose Calibration:**
- Hold each knee angle (0°, 30°, 60°, 90°) for 10s
- Record mean resistance value (Ω)
- Fit 2nd-order polynomial: `angle = a₀ + a₁·R + a₂·R²`

**Example (Left Knee):**
| Angle (°) | Resistance (Ω) |
|-----------|----------------|
| 0         | 11.05          |
| 30        | 11.4           |
| 60        | 11.7           |
| 90        | 13.3           |

Resulting polynomial enables real-time angle estimation from streaming resistance data.

## Validation Interpretation

**RMSE < 5°**: Excellent agreement (typical for calibrated systems)  
**RMSE 5–10°**: Good agreement (acceptable for gait monitoring)  
**RMSE > 10°**: Poor agreement (recalibrate or check hardware)

**Bland-Altman Limits of Agreement:**
- If mean difference ≈ 0° and LoA ±5°: System is unbiased with low variance
- If LoA > ±15°: Consider recalibration or sensor placement adjustment

**Correlation (r):**
- r > 0.95: Excellent tracking
- r > 0.85: Good tracking
- r < 0.80: Check sensor calibration

## Usage Example

```bash
# 1. Collect data
python Visualize_both_knees_EXPORT.py
# → Wear leggings, click Record, perform static poses + gait walk, click Stop

# 2. Merge asynchronous readings
python merge_strain_csv.py strain_data_20260422_161526.csv

# 3. Calibrate & generate synthetic IMU
python generate_synthetic_imu.py

# 4. Validate
python fusion_validation_pipeline.py
# → Review strain_imu_fusion_validation.png for metrics
```

## Integration with Real Xsens IMU Data

To validate with actual Awinda IMU data instead of synthetic:

1. Export gait trial from MT Manager as CSV with columns:
   - `timestamp_s`
   - `left_shank_accel_x`, `left_shank_accel_y`, `left_shank_accel_z`
   - `left_shank_gyro_x`, `left_shank_gyro_y`, `left_shank_gyro_z`
   - `right_shank_accel_x`, ... (same for right)

2. Save as `synthetic_imu_gait.csv` (reuse column naming)

3. Run `fusion_validation_pipeline.py` → obtains validation metrics against real IMU ground truth

## Dependencies

```
pandas
numpy
scipy
scikit-learn
matplotlib
bleak (for BLE streaming)
```

Install via:
```bash
pip install pandas numpy scipy scikit-learn matplotlib bleak
```

## Project Context

**Application**: Real-time gait monitoring for rehabilitation, sports analytics, or clinical assessment.

**Advantage over IMU-only**: 
- Strain sensors are lightweight, low-power, and integrate directly into fabric
- Cheaper than multi-IMU systems
- Validation ensures clinical-grade accuracy

**Next Steps**:
- Gait event detection (heel strike identification from gyro peaks)
- Time-normalization and ensemble waveform comparison
- Patient cohort validation
- Integration with wearable platform

## Authors & Citation

Research on machine-knitted multi-sensor garments for dynamic gait monitoring.

For questions or contributions, see project documentation.

## License

MIT
