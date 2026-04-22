import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error

# Load data
strain_calib = pd.read_csv('imu data (synthetic)/strain_angles_calibrated.csv')
imu_synthetic = pd.read_csv('imu data (synthetic)/synthetic_imu_gait.csv')

# Extract gait window (90-103.5s)
gait_mask = (strain_calib['timestamp_s'] >= 90) & (strain_calib['timestamp_s'] <= 103.5)
strain_gait = strain_calib[gait_mask].reset_index(drop=True)

print(f"Strain gait window: {len(strain_gait)} samples")
print(f"IMU synthetic gait: {len(imu_synthetic)} samples")


# === IMU Fusion: Integrate gyro to get angle ===
def integrate_gyro_to_angle(gyro_y_deg_s, timestamps, initial_angle_deg=0):
    """
    Integrate gyroscope (gyro_y = flexion axis) to estimate knee angle.
    Uses cumulative trapezoidal integration.
    """
    # Handle NaN by forward-fill then zero-fill
    gyro_clean = pd.Series(gyro_y_deg_s).ffill().fillna(0).values
    dt = np.diff(timestamps)
    angle = initial_angle_deg + np.concatenate([[0], np.cumsum(gyro_clean[:-1] * dt)])
    return angle


# Left knee
left_imu_angle = integrate_gyro_to_angle(
    imu_synthetic['left_shank_gyro_y'].values,
    imu_synthetic['timestamp_s'].values,
    initial_angle_deg=imu_synthetic['left_knee_angle_deg'].iloc[0]
)

# Right knee
right_imu_angle = integrate_gyro_to_angle(
    imu_synthetic['right_shank_gyro_y'].values,
    imu_synthetic['timestamp_s'].values,
    initial_angle_deg=imu_synthetic['right_knee_angle_deg'].iloc[0]
)

imu_synthetic['left_angle_integrated'] = left_imu_angle
imu_synthetic['right_angle_integrated'] = right_imu_angle

print("\nIMU angle integration complete.")

# === Synchronize & Compare ===
# Interpolate strain to IMU timestamps
from scipy.interpolate import interp1d

strain_interp_left = interp1d(
    strain_gait['timestamp_s'], strain_gait['left_angle_deg'],
    kind='linear', fill_value='extrapolate'
)
strain_interp_right = interp1d(
    strain_gait['timestamp_s'], strain_gait['right_angle_deg'],
    kind='linear', fill_value='extrapolate'
)

imu_synthetic['strain_left_interp'] = strain_interp_left(imu_synthetic['timestamp_s'])
imu_synthetic['strain_right_interp'] = strain_interp_right(imu_synthetic['timestamp_s'])


# === Metrics ===
def calc_metrics(truth, estimate, name):
    """Calculate RMSE, MAE, correlation, and offset."""
    rmse = np.sqrt(mean_squared_error(truth, estimate))
    mae = np.mean(np.abs(truth - estimate))
    corr = np.corrcoef(truth, estimate)[0, 1]

    # Bland-Altman: mean difference + limits of agreement
    mean_diff = np.mean(truth - estimate)
    std_diff = np.std(truth - estimate)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.3f}°")
    print(f"  MAE:  {mae:.3f}°")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Mean diff: {mean_diff:.3f}° (±{std_diff:.3f}°)")
    print(f"  LoA: [{loa_lower:.2f}°, {loa_upper:.2f}°]")

    return {
        'rmse': rmse, 'mae': mae, 'corr': corr,
        'mean_diff': mean_diff, 'std_diff': std_diff,
        'loa_upper': loa_upper, 'loa_lower': loa_lower
    }


metrics_left = calc_metrics(
    imu_synthetic['strain_left_interp'].values,
    imu_synthetic['left_angle_integrated'].values,
    'Left Knee (Strain vs IMU-Integrated)'
)

metrics_right = calc_metrics(
    imu_synthetic['strain_right_interp'].values,
    imu_synthetic['right_angle_integrated'].values,
    'Right Knee (Strain vs IMU-Integrated)'
)

# === Plots ===
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.patch.set_facecolor('#fafafa')

# --- Left Knee ---
# Comparison
ax = axes[0, 0]
ax.plot(imu_synthetic['timestamp_s'], imu_synthetic['strain_left_interp'],
        label='Strain (truth)', color='#185FA5', lw=2, alpha=0.8)
ax.plot(imu_synthetic['timestamp_s'], imu_synthetic['left_angle_integrated'],
        label='IMU (gyro integrated)', color='#FF6B6B', lw=2, alpha=0.6, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Knee Flexion (°)')
ax.set_title('Left Knee: Angle Comparison')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Bland-Altman
ax = axes[1, 0]
mean_vals = (imu_synthetic['strain_left_interp'] + imu_synthetic['left_angle_integrated']) / 2
diff_vals = imu_synthetic['strain_left_interp'] - imu_synthetic['left_angle_integrated']
ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20, color='#185FA5')
ax.axhline(metrics_left['mean_diff'], color='red', linestyle='-', lw=2, label='Mean diff')
ax.axhline(metrics_left['loa_upper'], color='red', linestyle='--', lw=1.5, label='LoA')
ax.axhline(metrics_left['loa_lower'], color='red', linestyle='--', lw=1.5)
ax.axhline(0, color='black', linestyle=':', lw=1)
ax.set_xlabel('Mean Angle (°)')
ax.set_ylabel('Difference (°)')
ax.set_title('Left Knee: Bland-Altman Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# Error histogram
ax = axes[2, 0]
error = imu_synthetic['strain_left_interp'] - imu_synthetic['left_angle_integrated']
ax.hist(error, bins=30, color='#185FA5', alpha=0.7, edgecolor='black')
ax.axvline(metrics_left['mean_diff'], color='red', linestyle='-', lw=2, label=f"μ={metrics_left['mean_diff']:.2f}°")
ax.set_xlabel('Error (Strain - IMU) (°)')
ax.set_ylabel('Frequency')
ax.set_title('Left Knee: Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# --- Right Knee ---
# Comparison
ax = axes[0, 1]
ax.plot(imu_synthetic['timestamp_s'], imu_synthetic['strain_right_interp'],
        label='Strain (truth)', color='#993C1D', lw=2, alpha=0.8)
ax.plot(imu_synthetic['timestamp_s'], imu_synthetic['right_angle_integrated'],
        label='IMU (gyro integrated)', color='#FFB347', lw=2, alpha=0.6, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Knee Flexion (°)')
ax.set_title('Right Knee: Angle Comparison')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Bland-Altman
ax = axes[1, 1]
mean_vals_r = (imu_synthetic['strain_right_interp'] + imu_synthetic['right_angle_integrated']) / 2
diff_vals_r = imu_synthetic['strain_right_interp'] - imu_synthetic['right_angle_integrated']
ax.scatter(mean_vals_r, diff_vals_r, alpha=0.5, s=20, color='#993C1D')
ax.axhline(metrics_right['mean_diff'], color='red', linestyle='-', lw=2, label='Mean diff')
ax.axhline(metrics_right['loa_upper'], color='red', linestyle='--', lw=1.5, label='LoA')
ax.axhline(metrics_right['loa_lower'], color='red', linestyle='--', lw=1.5)
ax.axhline(0, color='black', linestyle=':', lw=1)
ax.set_xlabel('Mean Angle (°)')
ax.set_ylabel('Difference (°)')
ax.set_title('Right Knee: Bland-Altman Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# Error histogram
ax = axes[2, 1]
error_r = imu_synthetic['strain_right_interp'] - imu_synthetic['right_angle_integrated']
ax.hist(error_r, bins=30, color='#993C1D', alpha=0.7, edgecolor='black')
ax.axvline(metrics_right['mean_diff'], color='red', linestyle='-', lw=2, label=f"μ={metrics_right['mean_diff']:.2f}°")
ax.set_xlabel('Error (Strain - IMU) (°)')
ax.set_ylabel('Frequency')
ax.set_title('Right Knee: Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('strain_imu_fusion_validation.png', dpi=150, bbox_inches='tight')
print("\n✓ Validation plot saved: strain_imu_fusion_validation.png")
plt.close()

# === Save Fused Data ===
fused = imu_synthetic[[
    'timestamp_s',
    'left_knee_angle_deg',
    'left_shank_accel_x', 'left_shank_accel_y', 'left_shank_accel_z',
    'left_shank_gyro_x', 'left_shank_gyro_y', 'left_shank_gyro_z',
    'right_knee_angle_deg',
    'right_shank_accel_x', 'right_shank_accel_y', 'right_shank_accel_z',
    'right_shank_gyro_x', 'right_shank_gyro_y', 'right_shank_gyro_z',
]].copy()

fused['left_angle_imu_integrated'] = left_imu_angle
fused['right_angle_imu_integrated'] = right_imu_angle
fused['left_angle_strain'] = imu_synthetic['strain_left_interp']
fused['right_angle_strain'] = imu_synthetic['strain_right_interp']

fused.to_csv('fused_strain_imu_gait.csv', index=False)
print("✓ Fused data saved: fused_strain_imu_gait.csv")
print("\nFusion pipeline complete. Ready for gait event detection & cycle analysis.")