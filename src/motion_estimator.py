import os
import pandas as pd
import numpy as np
import logging

class MotionEstimator:
    def __init__(self, input_csv, output_dir):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.output_csv = os.path.join(output_dir, "motion_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.pixels_per_meter = 10  # Calibration factor

    def estimate_movement(self):
        try:
            if not os.path.exists(self.input_csv):
                self.logger.error(f"Input CSV {self.input_csv} not found.")
                return False
            df = pd.read_csv(self.input_csv)
            if df.empty:
                self.logger.warning("Input CSV is empty.")
                return False
            df = df[df['class'].str.lower().isin(['car', 'police car'])]
            if df.empty:
                self.logger.warning("No car/police car detections in input CSV.")
                return False
            motion_data = []
            for track_id in df['track_id'].unique():
                track_df = df[df['track_id'] == track_id].sort_values('timestamp_ms')
                if len(track_df) < 2:
                    continue
                for i in range(1, len(track_df)):
                    prev_row = track_df.iloc[i-1]
                    curr_row = track_df.iloc[i]
                    time_diff = (curr_row['timestamp_ms'] - prev_row['timestamp_ms']) / 1000
                    if time_diff <= 0:
                        continue
                    prev_center = [(prev_row['x_min'] + prev_row['x_max']) / 2, (prev_row['y_min'] + prev_row['y_max']) / 2]
                    curr_center = [(curr_row['x_min'] + curr_row['x_max']) / 2, (curr_row['y_min'] + curr_row['y_max']) / 2]
                    dx = curr_center[0] - prev_center[0]
                    dy = curr_center[1] - prev_center[1]
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    distance_meters = distance_pixels / self.pixels_per_meter
                    speed = distance_meters / time_diff if time_diff > 0 else 0
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    direction = self.get_direction(angle)
                    motion_data.append({
                        'track_id': track_id,
                        'timestamp_ms': curr_row['timestamp_ms'],
                        'class': curr_row['class'],
                        'speed_m_s': speed,
                        'direction': direction,
                        'distance_m': distance_meters
                    })
            if motion_data:
                motion_df = pd.DataFrame(motion_data)
                motion_df.to_csv(self.output_csv, index=False)
                self.logger.info(f"Saved motion analysis to {self.output_csv} with {len(motion_df)} entries")
                return True
            else:
                self.logger.warning("No valid tracks for motion analysis.")
                pd.DataFrame(columns=['track_id', 'timestamp_ms', 'class', 'speed_m_s', 'direction', 'distance_m']).to_csv(self.output_csv, index=False)
                return False
        except Exception as e:
            self.logger.error(f"Error estimating motion: {e}")
            pd.DataFrame(columns=['track_id', 'timestamp_ms', 'class', 'speed_m_s', 'direction', 'distance_m']).to_csv(self.output_csv, index=False)
            return False

    def get_direction(self, angle):
        if -45 <= angle < 45:
            return 'Right'
        elif 45 <= angle < 135:
            return 'Down'
        elif 135 <= angle or angle < -135:
            return 'Left'
        else:
            return 'Up'