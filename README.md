# Real-Time Object Detection and Tracking with Audio and Motion Analysis 


This repository contains the source code for **Computer Vision Project 2** at AIMS Senegal, May 2025. The system detects and tracks urban objects (traffic lights, cars, police cars, persons, buses, trucks, bicycles, motorcycles) in one-minute videos, with siren-based filtering, motion analysis, and heatmaps. Built with **YOLOv11n**, **ByteTrack**, and **Streamlit**, it meets requirements for real-time processing, evaluation, and interactivity.

## Features
- **Detection**: YOLOv11n detects 8 classes.
- **Tracking**: ByteTrack assigns unique IDs.
- **Siren Filtering**: YAMNet filters car/police car detections (confidence > 0.5).
- **Motion Analysis**: Estimates vehicle direction and speed (10 pixels/meter).
- **Heatmaps**: Visualizes object locations.
- **Streamlit UI**: Video uploads, object selection, result visualization.
- **Evaluation**: Precision, recall, mAP@0.5, ID switches.
- **Error Handling**: Warns for missing detections/sirens.

## Repository Structure

/home/students/Documents/projet-aims/
├── src/
│   ├── detect_track.py
│   ├── web_interface.py
│   ├── heatmap_generator.py
│   ├── evaluate_metrics.py
│   ├── audio_filter.py
│   ├── motion_estimator.py
├── resources/models/yolo11n.pt
├── predictions/
├── requirements.txt
├── README.md
├── LICENSE
├── streamlit_ui.png
├── heatmap_car.png

streamlit run src/web_interface.py

Outputs (predictions/):
processed_video.mp4
position_log.csv
filtered_position_log.csv
motion_log.csv
heatmap_police_car.png, etc.
evaluation_report.txt
groundtruth_template.csv

# License

MIT License

Copyright (c) 2025 [cOMPUTER VISION GROUP 2]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
