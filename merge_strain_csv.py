import pandas as pd
import numpy as np
import sys


def merge_strain_data(csv_path):
    """
    Merge left/right strain readings by interpolating to a common timeline.

    Input: CSV with columns [timestamp_s, left_knee_ohms, right_knee_ohms]
           (one or both may be NaN per row)

    Output: merged_strain_data.csv with synchronized readings
    """
    df = pd.read_csv(csv_path)


    # Drop rows where both are NaN
    df = df.dropna(how='all', subset=['left_knee_ohms', 'right_knee_ohms'])

    # Separate left and right, drop NaNs within each
    left = df[['timestamp_s', 'left_knee_ohms']].dropna()
    right = df[['timestamp_s', 'right_knee_ohms']].dropna()

    # Create common timeline (union of both)
    common_times = sorted(set(left['timestamp_s'].values) | set(right['timestamp_s'].values))

    # Interpolate both to common timeline
    left_interp = np.interp(common_times, left['timestamp_s'].values, left['left_knee_ohms'].values)
    right_interp = np.interp(common_times, right['timestamp_s'].values, right['right_knee_ohms'].values)

    # Create merged dataframe
    merged = pd.DataFrame({
        'timestamp_s': common_times,
        'left_knee_ohms': left_interp,
        'right_knee_ohms': right_interp
    })

    # Save
    output_path = csv_path.replace('.csv', '_merged.csv')
    merged.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")
    print(f"Rows: {len(merged)}, both columns filled: Yes")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_strain_csv.py <strain_data.csv>")
        sys.exit(1)
    merge_strain_data(sys.argv[1])