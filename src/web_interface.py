import streamlit as st
import os
import cv2 as cv
import pandas as pd
import tempfile
import sys
import shutil
import time
import logging

# Add project directory to Python path
base_dir = "/home/students/Documents/projet-aims"
sys.path.append(os.path.join(base_dir, "src"))

try:
    from detect_track import Detector
    from evaluate_metrics import evaluate_detections
    from heatmap_generator import generate_heatmap
    from audio_filter import AudioFilter
    from motion_estimator import MotionEstimator
except ImportError as e:
    st.error(f"Module not found: {e}. Ensure detect_track.py, evaluate_metrics.py, heatmap_generator.py, audio_filter.py, and motion_estimator.py are in ~/Documents/projet-aims/src/.")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
PREDICTIONS_DIR = os.path.join(base_dir, "predictions")
OUTPUT_DIR = os.path.join(base_dir, "src", "outputs")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(255, 107, 107, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><defs><pattern id="grain" patternUnits="userSpaceOnUse" width="10" height="10"><circle cx="5" cy="5" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(#grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        position: relative;
        z-index: 2;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        position: relative;
        z-index: 2;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .results-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .config-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .config-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .upload-zone {
        border: 2px dashed rgba(255, 107, 107, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 107, 107, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def read_evaluation_results(output_file):
    try:
        if not os.path.exists(output_file):
            return None
        with open(output_file, "r") as f:
            lines = f.readlines()
        metrics = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                metrics[key.strip()] = value.strip()
        return metrics
    except Exception as e:
        logger.error(f"Error reading evaluation results: {e}")
        st.error(f"Error reading evaluation results: {e}")
        return None

def create_groundtruth_template(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
        if df.empty or len(df) <= 1:
            st.warning("position_log.csv is empty. Ground truth template will be empty.")
        df = df[["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max"]]
        df.to_csv(output_csv, index=False)
        return output_csv
    except Exception as e:
        logger.error(f"Error creating ground truth template: {e}")
        st.error(f"Error creating ground truth template: {e}")
        return None

def is_valid_csv(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        df = pd.read_csv(file_path)
        return not df.empty and len(df.columns) > 0
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False

def filter_csv_by_timestamps(predictions_csv, groundtruth_csv):
    try:
        pred_df = pd.read_csv(predictions_csv)
        gt_df = pd.read_csv(groundtruth_csv)
        common_timestamps = set(pred_df['timestamp_ms']).intersection(set(gt_df['timestamp_ms']))
        if not common_timestamps:
            st.warning("No common timestamps between predictions and ground truth. Evaluation may be incomplete.")
            return pred_df, gt_df
        pred_df = pred_df[pred_df['timestamp_ms'].isin(common_timestamps)]
        gt_df = gt_df[gt_df['timestamp_ms'].isin(common_timestamps)]
        return pred_df, gt_df
    except Exception as e:
        logger.error(f"Error filtering CSVs: {e}")
        st.error(f"Error filtering CSVs: {e}")
        return None, None

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Object Detector</h1>
        <p>Real-time detection and tracking of objects using YOLOv11 with Audio and Motion Analysis</p>
        <div style="margin-top: 1rem;">
            <span class="status-badge">YOLOv11 Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Model")
        model_info = {
            "description": "Pretrained YOLOv11n for object detection and tracking",
            "classes": ["traffic light", "police car", "car", "person", "bus", "truck", "bicycle", "motorcycle"],
            "parameters": "~2.6M"
        }
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div style="font-size: 1.5em;">🚗</div>
                <div><strong>{len(model_info['classes'])}</strong></div>
                <div style="font-size: 0.8em;">Classes</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5em;">⚡</div>
                <div><strong>{model_info['parameters']}</strong></div>
                <div style="font-size: 0.8em;">Parameters</div>
            </div>
        </div>
        <p><strong>Description:</strong> {model_info['description']}</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔍 Detection Settings")
        target_object = st.selectbox(
            "Target Object:",
            ["all"] + model_info["classes"],
            help="Select the object to detect (or 'all' for all classes)"
        )
        confidence = st.slider("Confidence Threshold", 0.01, 0.9, 0.01, step=0.01, help="Set to 0.01 for maximum detections")
        enable_audio_filter = st.checkbox("Enable Siren-Based Filtering", value=True, help="Filter car/police car detections based on siren sounds")
        enable_live_window = st.checkbox("Enable Live Detection Window", value=False, help="Open a live OpenCV window (requires GTK/X11 setup)")

        st.markdown("### 📊 Evaluation Settings")
        ground_truth_file = st.file_uploader(
            "Upload Ground Truth CSV",
            type=["csv"],
            help="CSV with columns: timestamp_ms, track_id, class, x_min, y_min, x_max, y_max. Download the template below."
        )
        iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.5, step=0.05)
        time_threshold = st.number_input("Time Threshold (ms)", 0, 1000, 100)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload & Process Video")
        
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload a video for detection",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv.VideoCapture(tfile.name)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="📹 Video Preview (First Frame)", use_container_width=True)
                    st.markdown(f"""
                    **Video Information:**
                    - Size: {int(cap.get(cv.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))} pixels
                    - FPS: {int(cap.get(cv.CAP_PROP_FPS))}
                    - Duration: {cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS):.2f} seconds
                    """)
                else:
                    st.warning("Could not read video frames. Check video format.")
                cap.release()
            else:
                st.error("Failed to open video file. Ensure it’s a valid MP4, AVI, or MOV.")

            if st.button("🚀 Process Video", type="primary"):
                processing_caption = "Video is processing, and a new window will open for live detection..." if enable_live_window else f"Video is processing with confidence threshold: {confidence:.2f} for {target_object}..."
                with st.spinner(processing_caption):
                    try:
                        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        position_log_path = os.path.join(PREDICTIONS_DIR, "position_log.csv")

                        target_objects = None if target_object == "all" else [target_object]
                        detector = Detector(
                            filepath=tfile.name,
                            target_objects=target_objects,
                            no_window=not enable_live_window,
                            evaluate=True
                        )
                        start_time = time.time()
                        try:
                            detector.process_video(confidence=confidence)
                        except Exception as e:
                            logger.error(f"Detector processing failed: {e}")
                            if enable_live_window:
                                st.warning("Live detection window failed to open. Ensure GTK/X11 is installed and DISPLAY is set (e.g., 'export DISPLAY=:0').")
                            raise e
                        st.info("Video processing completed.")
                        processing_time = time.time() - start_time
                        st.write(f"Processing time: {processing_time:.2f} seconds")

                        # Check detections
                        if is_valid_csv(position_log_path):
                            df = pd.read_csv(position_log_path)
                            car_count = len(df[df['class'].str.lower().isin(['car', 'police car'])])
                            st.success(f"Found {len(df)} total detections, including {car_count} car/police car detections.")
                            shutil.copy(position_log_path, os.path.join(OUTPUT_DIR, "position_log.csv"))
                        else:
                            st.warning(f"No valid detections in {position_log_path}. Ensure video contains cars/police cars, lower confidence, or check console logs.")

                        # Audio filtering
                        if enable_audio_filter and target_object in ["all", "car", "police car"] and is_valid_csv(position_log_path):
                            df = pd.read_csv(position_log_path)
                            if not df[df['class'].str.lower().isin(['car', 'police car'])].empty:
                                audio_filter = AudioFilter(tfile.name, output_dir=PREDICTIONS_DIR)
                                if audio_filter.extract_audio():
                                    if audio_filter.detect_sirens():
                                        audio_filter.filter_detections()
                                        st.info("Audio-based filtering completed.")
                                    else:
                                        st.warning("No sirens detected. Filtered log may be empty.")
                                    audio_filter.cleanup()
                                else:
                                    st.warning("Audio extraction failed. Skipping siren-based filtering.")
                            else:
                                st.warning("No car/police car detections to filter. Skipping audio analysis.")
                            filtered_log_path = os.path.join(PREDICTIONS_DIR, "filtered_position_log.csv")
                            if os.path.exists(filtered_log_path):
                                shutil.copy(filtered_log_path, os.path.join(OUTPUT_DIR, "filtered_position_log.csv"))
                        elif enable_audio_filter:
                            st.info("Skipping audio filtering: no valid detections or irrelevant target object.")

                        # Motion estimation
                        input_csv = os.path.join(PREDICTIONS_DIR, "filtered_position_log.csv" if enable_audio_filter and is_valid_csv(os.path.join(PREDICTIONS_DIR, "filtered_position_log.csv")) else "position_log.csv")
                        if is_valid_csv(input_csv):
                            df = pd.read_csv(input_csv)
                            if not df[df['class'].str.lower().isin(['car', 'police car'])].empty:
                                motion_estimator = MotionEstimator(input_csv, output_dir=PREDICTIONS_DIR)
                                if motion_estimator.estimate_movement():
                                    st.info("Motion estimation completed.")
                                else:
                                    st.warning("No valid car/police car tracks for motion analysis.")
                            else:
                                st.warning("No car/police car detections in input CSV. Skipping motion estimation.")
                        else:
                            st.warning(f"No valid input CSV at {input_csv}. Skipping motion estimation.")

                        # Processed video
                        processed_video_path = os.path.join(PREDICTIONS_DIR, "video_files", "processed_video.mp4")
                        if os.path.exists(processed_video_path):
                            st.success("Video processed successfully!")
                            with open(processed_video_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes, format="video/mp4")
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.error(f"Processed video not found at {processed_video_path}. Check permissions or console logs.")

                        # Heatmap
                        if is_valid_csv(position_log_path):
                            try:
                                df = pd.read_csv(position_log_path)
                                success = generate_heatmap()
                                if success:
                                    if target_object == "all":
                                        for class_name in df["class"].unique():
                                            heatmap_path = os.path.join(PREDICTIONS_DIR, f"heatmap_{class_name.replace(' ', '_')}.png")
                                            if os.path.exists(heatmap_path):
                                                st.image(heatmap_path, caption=f"📊 Heatmap: {class_name.capitalize()}", use_container_width=True)
                                                with open(heatmap_path, "rb") as f:
                                                    st.download_button(
                                                        label=f"📥 Download Heatmap ({class_name})",
                                                        data=f,
                                                        file_name=f"heatmap_{class_name}.png",
                                                        mime="image/png"
                                                    )
                                    else:
                                        heatmap_path = os.path.join(PREDICTIONS_DIR, f"heatmap_{target_object.replace(' ', '_')}.png")
                                        if os.path.exists(heatmap_path):
                                            st.image(heatmap_path, caption=f"📊 Heatmap: {target_object.capitalize()}", use_container_width=True)
                                            with open(heatmap_path, "rb") as f:
                                                st.download_button(
                                                    label=f"📥 Download Heatmap",
                                                    data=f,
                                                    file_name=f"heatmap_{target_object}.png",
                                                    mime="image/png"
                                                )
                                else:
                                    st.warning("Failed to generate heatmap. Check console output.")
                            except Exception as e:
                                logger.error(f"Error generating heatmap: {e}")
                                st.error(f"Error generating heatmap: {e}")
                        else:
                            st.warning("No valid detections for heatmap generation.")

                    except Exception as e:
                        logger.error(f"Error processing video: {e}")
                        st.error(f"Error processing video: {e}")
                    finally:
                        os.unlink(tfile.name)

    with col2:
        st.markdown("### 🎯 Detection & Evaluation Results")

        position_log_path = os.path.join(PREDICTIONS_DIR, "position_log.csv")
        if uploaded_video is not None:
            if is_valid_csv(position_log_path):
                try:
                    st.markdown("#### 📋 Original Detection Log")
                    df = pd.read_csv(position_log_path)
                    st.dataframe(df)
                    with open(position_log_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Original Detection Log",
                            data=f,
                            file_name="position_log.csv",
                            mime="text/csv"
                        )

                    filtered_log_path = os.path.join(PREDICTIONS_DIR, "filtered_position_log.csv")
                    if is_valid_csv(filtered_log_path):
                        st.markdown("#### 📋 Filtered Detection Log (Siren-Based)")
                        filtered_df = pd.read_csv(filtered_log_path)
                        st.dataframe(filtered_df)
                        if filtered_df.empty:
                            st.info("Filtered log is empty. Ensure video has siren sounds and car/police car detections.")
                        with open(filtered_log_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Filtered Detection Log",
                                data=f,
                                file_name="filtered_position_log.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No siren-based filtered detections available. Ensure video has siren sounds and car/police car detections.")

                    motion_log_path = os.path.join(PREDICTIONS_DIR, "motion_log.csv")
                    if is_valid_csv(motion_log_path):
                        st.markdown("#### 🚗 Motion Analysis")
                        motion_df = pd.read_csv(motion_log_path)
                        st.dataframe(motion_df)
                        if motion_df.empty:
                            st.info("Motion log is empty. Ensure car/police car tracks are detected.")
                        with open(motion_log_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Motion Log",
                                data=f,
                                file_name="motion_log.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No motion analysis data available. Ensure car/police car tracks are detected.")

                    template_path = os.path.join(PREDICTIONS_DIR, "groundtruth_template.csv")
                    if create_groundtruth_template(position_log_path, template_path):
                        with open(template_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Ground Truth Template",
                                data=f,
                                file_name="groundtruth_template.csv",
                                mime="text/csv",
                                help="Download this template, verify/edit detections, and upload as ground truth CSV."
                            )

                    if ground_truth_file is not None:
                        gt_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        gt_tfile.write(ground_truth_file.read())
                        gt_tfile.close()

                        with st.spinner("Evaluating detections..."):
                            try:
                                eval_input_csv = filtered_log_path if is_valid_csv(filtered_log_path) else position_log_path
                                pred_df, gt_df = filter_csv_by_timestamps(eval_input_csv, gt_tfile.name)
                                if pred_df is not None and gt_df is not None:
                                    pred_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                                    gt_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                                    pred_df.to_csv(pred_temp.name, index=False)
                                    gt_df.to_csv(gt_temp.name, index=False)

                                    success = evaluate_detections(
                                        ground_truth_file=gt_temp.name,
                                        predictions_file=pred_temp.name,
                                        output_file=os.path.join(PREDICTIONS_DIR, "evaluation_report.txt"),
                                        iou_threshold=iou_threshold,
                                        time_threshold=time_threshold
                                    )
                                    os.unlink(pred_temp.name)
                                    os.unlink(gt_temp.name)

                                    if success:
                                        metrics = read_evaluation_results(os.path.join(PREDICTIONS_DIR, "evaluation_report.txt"))
                                        if metrics:
                                            st.markdown("""
                                            <div class="results-box">
                                                <h2>📊 Evaluation Results</h2>
                                                <div class="metrics-grid">
                                            """, unsafe_allow_html=True)
                                            for key, value in metrics.items():
                                                st.markdown(f"""
                                                <div class="metric-card">
                                                    <div style="font-size: 1.5em;">📈</div>
                                                    <div><strong>{value}</strong></div>
                                                    <div style="font-size: 0.8em;">{key}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            st.markdown("</div></div>", unsafe_allow_html=True)

                                            metrics_path = os.path.join(PREDICTIONS_DIR, "metrics.txt")
                                            if os.path.exists(metrics_path):
                                                with open(metrics_path, "r") as f:
                                                    st.markdown("#### 📈 Additional Metrics")
                                                    st.text(f.read())
                                                with open(metrics_path, "rb") as f:
                                                    st.download_button(
                                                        label="📥 Download Metrics",
                                                        data=f,
                                                        file_name="metrics.txt",
                                                        mime="text/plain"
                                                    )
                                        else:
                                            st.error("Evaluation results not found.")
                                    else:
                                        st.error("Evaluation failed. Check ground truth and prediction files.")
                                else:
                                    st.error("Failed to filter CSVs for evaluation.")
                            except Exception as e:
                                logger.error(f"Error evaluating detections: {e}")
                                st.error(f"Error evaluating detections: {e}")
                            finally:
                                os.unlink(gt_tfile.name)
                except Exception as e:
                    logger.error(f"Error loading detection results: {e}")
                    st.error(f"Error loading detection results: {e}")
            else:
                st.warning("No valid detections found in position_log.csv. Use a video with visible cars/police cars and set confidence to 0.01.")
        else:
            st.info("👆 Upload a video and process it to see detection results.")

        st.markdown("""
        ### 💡 Tips for Best Results
        
        **Recommended Video Format:**
        - Formats: MP4, AVI, MOV
        - Duration: At least 1 minute
        - Clear visibility of cars/police cars
        - Siren sounds for audio filtering
        - Well-lit, minimal occlusion
        
        **Ground Truth CSV:**
        - Columns: timestamp_ms, track_id, class, x_min, y_min, x_max, y_max
        - Classes: traffic light, police car, car, person, bus, truck, bicycle, motorcycle
        - Use the template provided
        
        **Model Settings:**
        - Model: YOLOv11n at resources/models/yolo11n.pt
        - Outputs: predictions/position_log.csv, filtered_position_log.csv, motion_log.csv
        - Confidence: Use 0.01 for testing, increase if too many false positives
        - Siren Filtering: Requires clear siren audio
        - Motion Analysis: Needs multiple frames per car track
        - Live Window: Requires libgtk-3-dev and DISPLAY=:0 (run 'export DISPLAY=:0' before Streamlit)
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(255, 107, 107, 0.1); border-radius: 15px; margin-top: 2rem;">
        <h4>🚗 Object Detection Project</h4>
        <p>Powered by YOLOv11 and Streamlit</p>
        <p style="font-size: 0.9em; opacity: 0.7;">
            Real-time detection and tracking • Audio-based filtering • Motion analysis • Heatmap visualization • Performance evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()