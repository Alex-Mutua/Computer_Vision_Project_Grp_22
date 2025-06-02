# audio_filter.py
import moviepy.editor as mp
import librosa
import numpy as np
import pandas as pd
import os
import tempfile
import tensorflow_hub as hub
import tensorflow as tf

class AudioFilter:
    def __init__(self, video_path, output_dir="Documents/projet-aims/predictions"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.position_log_path = os.path.join(output_dir, "position_log.csv")
        self.filtered_log_path = os.path.join(output_dir, "filtered_position_log.csv")
        self.audio_path = None
        self.siren_timestamps = []
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_map = {482: 'Siren'}

    def extract_audio(self):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                self.audio_path = audio_file.name
            video = mp.VideoFileClip(self.video_path)
            if video.audio is None:
                print("No audio track in video")
                return False
            video.audio.write_audiofile(self.audio_path)
            video.close()
            print(f"Audio extracted to {self.audio_path}")
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def detect_sirens(self, time_window=1000):
        try:
            if not self.audio_path or not os.path.exists(self.audio_path):
                print("No audio file available for siren detection")
                return False
            y, sr = librosa.load(self.audio_path, sr=16000)
            frame_length = 0.96
            hop_length = 0.48
            hop_samples = int(hop_length * sr)

            for i in range(0, len(y) - int(frame_length * sr), hop_samples):
                frame = y[i:i + int(frame_length * sr)]
                if len(frame) != int(frame_length * sr):
                    continue
                scores, _, _ = self.yamnet_model(frame)
                scores = tf.reduce_mean(scores, axis=0).numpy()
                if 482 in self.class_map and scores[482] > 0.5:
                    timestamp_ms = (i / sr) * 1000
                    self.siren_timestamps.append(timestamp_ms)

            self.siren_timestamps = sorted(set([int(t // time_window) * time_window for t in self.siren_timestamps]))
            print(f"Siren timestamps: {self.siren_timestamps}")
            return bool(self.siren_timestamps)
        except Exception as e:
            print(f"Error detecting sirens: {e}")
            return False

    def filter_detections(self, time_window=1000):
        try:
            if not os.path.exists(self.position_log_path):
                print(f"Position log not found at {self.position_log_path}")
                # Write empty CSV with headers
                pd.DataFrame(columns=["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max", "confidence"]).to_csv(self.filtered_log_path, index=False)
                return False

            df = pd.read_csv(self.position_log_path)
            if df.empty:
                print("Position log is empty")
                df.to_csv(self.filtered_log_path, index=False)
                return False

            non_car_df = df[~df['class'].str.lower().isin(['car', 'police car'])].copy()
            car_df = df[df['class'].str.lower().isin(['car', 'police car'])].copy()
            if car_df.empty:
                print("No car or police car detections to filter")
                non_car_df.to_csv(self.filtered_log_path, index=False)
                return True

            if not self.siren_timestamps:
                print("No sirens detected; keeping only non-car detections")
                non_car_df.to_csv(self.filtered_log_path, index=False)
                return True

            car_df['time_window'] = (car_df['timestamp_ms'] // time_window).astype(int) * time_window
            siren_windows = set(self.siren_timestamps)
            filtered_car_df = car_df[car_df['time_window'].isin(siren_windows)]
            filtered_car_df = filtered_car_df.drop(columns=['time_window'])
            filtered_df = pd.concat([non_car_df, filtered_car_df], ignore_index=True)
            filtered_df.to_csv(self.filtered_log_path, index=False)
            print(f"Retained {len(filtered_car_df)} car/police car detections after siren filtering")
            return True
        except Exception as e:
            print(f"Error filtering detections: {e}")
            # Write empty CSV with headers
            pd.DataFrame(columns=["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max", "confidence"]).to_csv(self.filtered_log_path, index=False)
            return False

    def cleanup(self):
        if self.audio_path and os.path.exists(self.audio_path):
            os.unlink(self.audio_path)
            print(f"Cleaned up {self.audio_path}")