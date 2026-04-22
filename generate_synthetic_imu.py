import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Load strain data
strain_df = pd.read_csv('merged collected data/(good_wed)strain_data_20260422_161526_merged.csv')

# Calibration: strain (Ω) -> knee flexion angle (°)
# Standing (0°): 55 Ω, 30°: 65 Ω, 60°: 70 Ω, 90°: 80 Ω
# Left knee calibration
left_calib_ohms = np.array([11.05, 11.4, 11.7, 13.3])
left_calib_angles = np.array([0, 30, 60, 90])
left_poly = np.polyfit(left_calib_ohms, left_calib_angles, 2)

# Right knee calibration
right_calib_ohms = np.array([15.39, 15.27, 15.84, 24.1])
right_calib_angles = np.array([0, 30, 60, 90])
right_poly = np.polyfit(right_calib_ohms, right_calib_angles, 2)


# Convert strain to angle
def strain_to_angle(ohms, poly):
    return np.polyval(poly, ohms)


strain_df['left_angle_deg'] = strain_to_angle(strain_df['left_knee_ohms'], left_poly)
strain_df['right_angle_deg'] = strain_to_angle(strain_df['right_knee_ohms'], right_poly)

# Clamp angles to [0, 120]
strain_df['left_angle_deg'] = strain_df['left_angle_deg'].clip(0, 120)
strain_df['right_angle_deg'] = strain_df['right_angle_deg'].clip(0, 120)

print("Strain-to-Angle Calibration:")
print(f"Left knee: {left_poly}")
print(f"Right knee: {right_poly}")
print(f"\nSample angles:")
print(strain_df[['timestamp_s', 'left_knee_ohms', 'left_angle_deg', 'right_knee_ohms', 'right_angle_deg']].head(20))

# Extract gait walk data (timestamps 90-100s, adjust if needed)
gait_mask = (strain_df['timestamp_s'] >= 90) & (strain_df['timestamp_s'] <= 103.5)
gait_df = strain_df[gait_mask].reset_index(drop=True)

print(f"\nGait section: {len(gait_df)} samples, {gait_df['timestamp_s'].min():.1f}–{gait_df['timestamp_s'].max():.1f}s")


# Generate synthetic IMU data for gait section
# Thigh IMU: oriented along femur, captures flexion in gyro Y-axis
# Shank IMU (side): oriented along tibia, captures flexion in gyro Y-axis

def generate_imu_from_angle(angle_deg, angle_vel_deg_s, noise_level=0.5):
    """
    Generate synthetic IMU (accel, gyro, mag) from knee angle.

    Args:
        angle_deg: knee flexion angle (0-120°)
        angle_vel_deg_s: angular velocity (°/s)
        noise_level: Gaussian noise std dev

    Returns:
        dict with accel (m/s²), gyro (°/s), mag (normalized)
    """
    # Gravity vector
    g = 9.81

    # Shank: tilted by angle relative to vertical
    # Thigh: tilted by (90° - angle) relative to vertical (assumes torso upright)

    # Accelerometer (gravity in sensor frame)
    # Shank IMU on side of shin: gravity mostly in X, flexion causes Y component
    shank_accel_y = g * np.sin(np.radians(angle_deg))
    shank_accel_z = g * np.cos(np.radians(angle_deg))
    shank_accel_x = np.random.normal(0, noise_level)

    # Thigh IMU on lateral surface
    thigh_accel_y = g * np.sin(np.radians(90 - angle_deg))
    thigh_accel_z = g * np.cos(np.radians(90 - angle_deg))
    thigh_accel_x = np.random.normal(0, noise_level)

    # Gyroscope (angular velocity)
    # Flexion is rotation about medial-lateral (Y) axis
    gyro_y = angle_vel_deg_s
    gyro_x = np.random.normal(0, noise_level)
    gyro_z = np.random.normal(0, noise_level)

    # Magnetometer (simplified: Earth's field ~50 µT, rotates with knee)
    mag_x = 50 * np.cos(np.radians(angle_deg))
    mag_y = np.random.normal(0, 5)
    mag_z = 50 * np.sin(np.radians(angle_deg))

    return {
        'accel_x': shank_accel_x, 'accel_y': shank_accel_y, 'accel_z': shank_accel_z,
        'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
        'mag_x': mag_x, 'mag_y': mag_y, 'mag_z': mag_z,
    }


# Compute angular velocity via finite difference
gait_df['left_angle_vel'] = gait_df['left_angle_deg'].diff() / gait_df['timestamp_s'].diff() * 1.0
gait_df['right_angle_vel'] = gait_df['right_angle_deg'].diff() / gait_df['timestamp_s'].diff() * 1.0
gait_df['left_angle_vel'].fillna(0, inplace=True)
gait_df['right_angle_vel'].fillna(0, inplace=True)

# Generate IMU data
imu_data = []
for idx, row in gait_df.iterrows():
    timestamp = row['timestamp_s']
    left_angle = row['left_angle_deg']
    left_vel = row['left_angle_vel']
    right_angle = row['right_angle_deg']
    right_vel = row['right_angle_vel']

    # Left shank IMU
    left_imu = generate_imu_from_angle(left_angle, left_vel)

    # Right shank IMU
    right_imu = generate_imu_from_angle(right_angle, right_vel)

    # Combine
    imu_data.append({
        'timestamp_s': timestamp,
        'left_knee_angle_deg': left_angle,
        'left_shank_accel_x': left_imu['accel_x'],
        'left_shank_accel_y': left_imu['accel_y'],
        'left_shank_accel_z': left_imu['accel_z'],
        'left_shank_gyro_x': left_imu['gyro_x'],
        'left_shank_gyro_y': left_imu['gyro_y'],
        'left_shank_gyro_z': left_imu['gyro_z'],
        'right_knee_angle_deg': right_angle,
        'right_shank_accel_x': right_imu['accel_x'],
        'right_shank_accel_y': right_imu['accel_y'],
        'right_shank_accel_z': right_imu['accel_z'],
        'right_shank_gyro_x': right_imu['gyro_x'],
        'right_shank_gyro_y': right_imu['gyro_y'],
        'right_shank_gyro_z': right_imu['gyro_z'],
    })

imu_df = pd.DataFrame(imu_data)

# Save synthetic IMU
imu_df.to_csv('synthetic_imu_gait.csv', index=False)
print(f"\nSynthetic IMU data saved: synthetic_imu_gait.csv")

# Save strain angles (full dataset)
strain_df[['timestamp_s', 'left_knee_ohms', 'left_angle_deg', 'right_knee_ohms', 'right_angle_deg']].to_csv(
    'strain_angles_calibrated.csv', index=False)
print(f"Calibrated strain angles saved: strain_angles_calibrated.csv")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Full session
axes[0].plot(strain_df['timestamp_s'], strain_df['left_angle_deg'], label='Left Knee', color='#185FA5', lw=1.5)
axes[0].plot(strain_df['timestamp_s'], strain_df['right_angle_deg'], label='Right Knee', color='#993C1D', lw=1.5)
axes[0].axvline(90, color='green', linestyle='--', alpha=0.5, label='Gait start')
axes[0].axvline(103.5, color='red', linestyle='--', alpha=0.5, label='Gait end')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Knee Flexion (°)')
axes[0].set_title('Calibrated Strain Sensor Data')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gait zoom
axes[1].plot(imu_df['timestamp_s'], imu_df['left_knee_angle_deg'], label='Left (IMU-gen)', color='#185FA5', lw=1.5)
axes[1].plot(imu_df['timestamp_s'], imu_df['right_knee_angle_deg'], label='Right (IMU-gen)', color='#993C1D', lw=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Knee Flexion (°)')
axes[1].set_title('Synthetic IMU-Derived Angles (Gait Walk)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strain_to_imu_validation.png', dpi=150)
print(f"Validation plot saved: strain_to_imu_validation.png")
plt.close()

print("\n✓ Ready for cross-validation pipeline.")