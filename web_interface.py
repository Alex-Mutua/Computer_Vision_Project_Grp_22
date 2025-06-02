import streamlit as st
import os
import cv2 as cv
import pandas as pd
import tempfile
import sys
import os
import tempfile

temp_dir = tempfile.gettempdir()
video_path = os.path.join(temp_dir, "uploaded_video.mp4")


# Add project directory to Python path
base_dir = "/home/students/Documents/projet-aims"
sys.path.append(os.path.join(base_dir, "src"))

try:
    from detect_track import Detector
    from evaluate_metrics import evaluate_detections
    from heatmap_generator import generate_heatmap
except ImportError as e:
    st.error(f"Module not found: {e}. Ensure detect_track.py, evaluate_metrics.py, and heatmap_generator.py are in ~/Documents/projet-aims/src/.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🚨 Object Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Output directory
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
    """Read evaluation metrics from the output file."""
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
        st.error(f"Error reading evaluation results: {e}")
        return None

def create_groundtruth_template(input_csv, output_csv):
    """Create a ground truth CSV template from position_log.csv."""
    try:
        df = pd.read_csv(input_csv)
        if df.empty or len(df) <= 1:
            st.warning("position_log.csv is empty. Ground truth template will be empty.")
        df = df[["timestamp_ms", "class", "x_min", "y_min", "x_max", "y_max"]]
        df.to_csv(output_csv, index=False)
        return output_csv
    except Exception as e:
        st.error(f"Error creating ground truth template: {e}")
        return None

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚨 Object Detector</h1>
        <p>Real-time detection and tracking of objects using YOLOv11</p>
        <div style="margin-top: 1rem;">
            <span class="status-badge">YOLOv11 Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model configuration
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

        # Detection parameters
        st.markdown("### 🔍 Detection Settings")
        target_object = st.selectbox(
            "Target Object:",
            ["all"] + model_info["classes"],
            help="Select the object to detect (or 'all' for all classes)"
        )
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, step=0.05)

        # Evaluation parameters
        st.markdown("### 📊 Evaluation Settings")
        ground_truth_file = st.file_uploader(
            "Upload Ground Truth CSV",
            type=["csv"],
            help="CSV with columns: timestamp_ms, class, x_min, y_min, x_max, y_max. Download the template below to create one."
        )
        iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.5, step=0.05)
        time_threshold = st.number_input("Time Threshold (ms)", 0, 1000, 100)

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload & Process Video")
        
        # Upload zone
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload a video for detection",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_video is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            # Video preview (first frame)
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
                cap.release()

            # Process button
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video with YOLOv11..."):
                    try:
                        # Ensure output directory exists
                        os.makedirs(OUTPUT_DIR, exist_ok=True)

                        # Convert target_object to target_objects for Detector
                        target_objects = None if target_object == "all" else [target_object]

                        # Initialize and run detector
                        detector = Detector(
                            filepath=tfile.name,
                            target_objects=target_objects,
                            no_window=False,  # Enable OpenCV window
                            evaluate=True
                        )
                        st.info("Processing video... An OpenCV window will show real-time detections.")
                        detector.process_video(confidence=confidence)
                        st.info("Video processing completed.")

                        # Display processed video
                        processed_video_path = os.path.join(OUTPUT_DIR, "processed_video.avi")
                        if os.path.exists(processed_video_path):
                            st.success("Video processed successfully!")
                            with open(processed_video_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes, format="video/avi")
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name="processed_video.avi",
                                mime="video/avi"
                            )
                        else:
                            st.error(f"Processed video not found at {processed_video_path}. Check if src/outputs/ is writable.")

                        # Display heatmap
                        position_log_path = os.path.join(OUTPUT_DIR, "position_log.csv")
                        if os.path.exists(position_log_path):
                            try:
                                df = pd.read_csv(position_log_path)
                                if df.empty or len(df) <= 1:
                                    st.error("No detections in position_log.csv. Try lowering confidence or using a different video.")
                                else:
                                    success = generate_heatmap()
                                    if success:
                                        if target_object == "all":
                                            # Show per-class heatmaps
                                            for class_name in df["class"].unique():
                                                heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_{class_name.replace(' ', '_')}.png")
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
                                            # Show single heatmap
                                            heatmap_path = os.path.join(OUTPUT_DIR, "heatmap.png")
                                            if os.path.exists(heatmap_path):
                                                st.image(heatmap_path, caption=f"📊 Heatmap: {target_object.capitalize()}", use_container_width=True)
                                                with open(heatmap_path, "rb") as f:
                                                    st.download_button(
                                                        label=f"📥 Download Heatmap",
                                                        data=f,
                                                        file_name="heatmap.png",
                                                        mime="image/png"
                                                    )
                                    else:
                                        st.error("Failed to generate heatmap. Check console output and position_log.csv.")
                            except Exception as e:
                                st.error(f"Error generating heatmap: {e}")
                        else:
                            st.error(f"Position log not found at: {position_log_path}")

                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    finally:
                        os.unlink(tfile.name)

    with col2:
        st.markdown("### 🎯 Detection & Evaluation Results")

        position_log_path = os.path.join(OUTPUT_DIR, "position_log.csv")
        if uploaded_video is not None and os.path.exists(position_log_path):
            try:
                # Display detection log
                st.markdown("#### 📋 Detection Log")
                df = pd.read_csv(position_log_path)
                st.dataframe(df)
                with open(position_log_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Detection Log",
                        data=f,
                        file_name="position_log.csv",
                        mime="text/csv"
                    )

                # Create and offer ground truth template
                template_path = os.path.join(OUTPUT_DIR, "groundtruth_template.csv")
                if create_groundtruth_template(position_log_path, template_path):
                    with open(template_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Ground Truth Template",
                            data=f,
                            file_name="groundtruth_template.csv",
                            mime="text/csv",
                            help="Download this template, verify/edit detections, and upload as ground truth CSV."
                        )

                # Run evaluation if ground truth is provided
                if ground_truth_file is not None:
                    # Save ground truth to temporary file
                    gt_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    gt_tfile.write(ground_truth_file.read())
                    gt_tfile.close()

                    with st.spinner("Evaluating detections..."):
                        try:
                            success = evaluate_detections(
                                ground_truth_file=gt_tfile.name,
                                predictions_file=position_log_path,
                                output_file=os.path.join(OUTPUT_DIR, "evaluation_report.txt"),
                                iou_threshold=iou_threshold,
                                time_threshold=time_threshold
                            )
                            if success:
                                metrics = read_evaluation_results(os.path.join(OUTPUT_DIR, "evaluation_report.txt"))
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

                                    # Check for metrics.txt
                                    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
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
                                    st.error("Evaluation results not found")
                            else:
                                st.error("Evaluation failed. Check ground truth and prediction files.")
                        except Exception as e:
                            st.error(f"Error evaluating detections: {e}")
                        finally:
                            os.unlink(gt_tfile.name)
            except Exception as e:
                st.error(f"Error loading detection results: {e}")
        else:
            st.info("👆 Upload a video and process it to see detection results.")

        # Usage tips
        st.markdown("""
        ### 💡 Tips for Best Results
        
        **Recommended Video Format:**
        - Formats: MP4, AVI, MOV
        - Clear visibility of objects
        - Well-lit scenes with minimal occlusion
        
        **Ground Truth CSV:**
        - Columns: timestamp_ms, class, x_min, y_min, x_max, y_max
        - Ensure timestamps align with video frames
        - Classes must match: traffic light, police car, car, person, bus, truck, bicycle, motorcycle
        - Download the ground truth template, verify/edit detections, and upload
        
        **Model Settings:**
        - Uses YOLOv11n model at resources/models/yolo11n.pt
        - Outputs saved to src/outputs/
        - Adjust confidence and IoU thresholds for optimal detection
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(255, 107, 107, 0.1); border-radius: 15px; margin-top: 2rem;">
        <h4>🚗 Object Detection Project</h4>
        <p>Powered by YOLOv11 and Streamlit</p>
        <p style="font-size: 0.9em; opacity: 0.7;">
            Real-time detection and tracking • Heatmap visualization • Performance evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()