# process_with_audio.py
from detect_track import Detector
from audio_filter import AudioFilter
import os

def process_video_with_audio(video_path, target_objects, confidence=0.01, no_window=False, evaluate=True):
    # Run original detection
    detector = Detector(video_path, target_objects, no_window, evaluate)
    detector.process_video(confidence=confidence)

    # Apply audio filtering
    audio_filter = AudioFilter(video_path)
    if audio_filter.extract_audio():
        audio_filter.detect_sirens()
        audio_filter.filter_detections()
        audio_filter.cleanup()

if __name__ == "__main__":
    video_path = "/path/to/video.mp4"
    target_objects = ["car", "police car"]
    process_video_with_audio(video_path, target_objects)