# detect_track.py (corrected to match original functionality with minimal optimizations)
import cv2 as cv
import csv
import argparse
import os
from ultralytics import YOLO
import motmetrics as mm
import numpy as np
import pandas as pd
from evaluate_metrics import evaluate_detections
from heatmap_generator import generate_heatmap
from sklearn.metrics import precision_recall_fscore_support
import time

class Detector:
    def __init__(self, filepath, target_objects, no_window=True, evaluate=False):
        self.filepath = filepath
        self.target_objects = [obj.lower() for obj in target_objects] if target_objects else ["all"]
        self.model_path = "resources/models/yolo11n.pt"
        self.model = YOLO(self.model_path)
        self.no_window = no_window
        self.evaluate = evaluate
        self.base_dir = "/home/students/Documents/projet-aims"

        # Valid classes
        self.valid_classes = [
            "traffic light", "police car", "car",
            "person", "bus", "truck", "bicycle", "motorcycle"
        ]
        print("Model classes:", self.model.names)

        # Ensure output directories exist
        self.output_dir = os.path.join(self.base_dir, "src", "outputs")
        self.video_dir = self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, confidence=0.01):
        start_total = time.time()
        cap = cv.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Erreur : impossible de lire la vidéo {self.filepath}")

        w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
            print("Warning: Invalid FPS detected. Using fallback FPS=30.")

        output_video_path = os.path.join(self.output_dir, "processed_video.mp4")
        video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        if not video_writer.isOpened():
            print(f"Error: Failed to initialize VideoWriter for {output_video_path}. Trying XVID codec.")
            video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'XVID'), fps, (w, h))
            if not video_writer.isOpened():
                raise RuntimeError(f"Error: Cannot open VideoWriter for {output_video_path}.")

        if not self.no_window:
            try:
                cv.namedWindow("Détection et Tracking", cv.WINDOW_NORMAL)
                print("OpenCV: Window created")
            except cv.error as e:
                print(f"OpenCV Error: Failed to create window: {e}")

        output_csv_path = os.path.join(self.output_dir, "position_log.csv")
        detection_count = 0
        predictions = []

        try:
            with open(output_csv_path, "w", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max", "confidence"])

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Fin de la vidéo après {frame_count} frames")
                        break

                    input_frame = cv.resize(frame, (640, 640))
                    start_inference = time.time()
                    results = self.model.track(input_frame, persist=True, verbose=False, conf=confidence)[0]
                    end_inference = time.time()
                    print(f"Frame {frame_count}: Inference time: {end_inference - start_inference:.2f}s")

                    start_post = time.time()
                    annotated_frame = frame.copy()
                    timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
                    detected = False
                    frame_preds = []

                    if results.boxes is not None:
                        all_detected = []
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id].lower()
                            conf = box.conf[0].item()
                            all_detected.append(f"{class_name} (conf: {conf:.2f})")
                        print(f"Frame {frame_count}: Detected classes: {all_detected}")

                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id].lower()
                            if class_name in self.valid_classes and (self.target_objects == ["all"] or class_name in self.target_objects):
                                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                                x_min = int(x_min * w / 640)
                                y_min = int(y_min * h / 640)
                                x_max = int(x_max * w / 640)
                                y_max = int(y_max * h / 640)
                                track_id = int(box.id[0]) if box.id is not None else -1
                                conf = box.conf[0].item()

                                cv.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                label = f"ID: {track_id} {class_name} {conf:.2f}"
                                cv.putText(annotated_frame, label, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                writer.writerow([timestamp, track_id, class_name, x_min, y_min, x_max, y_max, conf])
                                detection_count += 1
                                detected = True

                                frame_preds.append({
                                    'id': track_id,
                                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                                    'conf': conf,
                                    'class': class_name
                                })

                    predictions.append(frame_preds)

                    if detected:
                        cv.putText(annotated_frame, f"{', '.join(self.target_objects).capitalize()} détecté !", (50, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv.putText(annotated_frame, f"Aucun {', '.join(self.target_objects)} détecté", (50, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    video_writer.write(annotated_frame)

                    if not self.no_window:
                        try:
                            cv.imshow("Détection et Tracking", annotated_frame)
                            if cv.waitKey(1) & 0xFF == ord('q'):
                                print("Arrêt par l'utilisateur.")
                                break
                        except cv.error as e:
                            print(f"OpenCV Error: Failed to display frame: {e}")

                    end_post = time.time()
                    print(f"Frame {frame_count}: Post-processing time: {end_post - start_post:.2f}s")
                    frame_count += 1

        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        finally:
            cap.release()
            video_writer.release()
            print(f"Video writer released for {output_video_path}")
            if not self.no_window:
                try:
                    cv.destroyAllWindows()
                    print("OpenCV: All windows closed")
                except cv.error as e:
                    print(f"OpenCV Error: Failed to close windows: {e}")

        print(f"Detections written to {output_csv_path}: {detection_count} total detections")
        if detection_count == 0:
            print("Warning: No objects detected. Check video content, target_objects, or confidence threshold.")

        end_total = time.time()
        print(f"Total processing time: {end_total - start_total:.2f}s")

        if detection_count > 0:
            generate_heatmap()
        if self.evaluate:
            self.compute_metrics(predictions)

    def compute_metrics(self, predictions):
        metrics_file = os.path.join(self.output_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            gt_file = os.path.join(self.output_dir, "groundtruth_template.csv")
            pred_file = os.path.join(self.output_dir, "position_log.csv")
            eval_report = os.path.join(self.output_dir, "evaluation_report.txt")
            if os.path.exists(gt_file):
                try:
                    evaluate_detections(gt_file, pred_file, eval_report)
                    with open(eval_report, "r") as report:
                        f.write("Custom Evaluation Metrics:\n")
                        f.write(report.read())
                except Exception as e:
                    print(f"Custom evaluation failed: {e}")
                    f.write(f"Custom Evaluation Metrics: Failed: {e}\n")
            else:
                print(f"Ground truth file {gt_file} not found. Skipping custom evaluation.")
                f.write("Custom Evaluation Metrics: Skipped (ground truth missing)\n")

            f.write("\nDetection Metrics:\n")
            try:
                pred_df = pd.read_csv(pred_file)
                if pred_df.empty:
                    print("No predictions in position_log.csv. Skipping detection metrics.")
                    f.write("Precision: N/A (no predictions)\n")
                    f.write("Recall: N/A (no predictions)\n")
                    f.write("mAP@0.5: N/A (no predictions)\n")
                elif os.path.exists(gt_file):
                    gt_df = pd.read_csv(gt_file)
                    if gt_df.empty:
                        print("Ground truth is empty. Skipping detection metrics.")
                        f.write("Precision: N/A (empty ground truth)\n")
                        f.write("Recall: N/A (empty ground truth)\n")
                        f.write("mAP@0.5: N/A (empty ground truth)\n")
                    else:
                        y_true = gt_df["class"].values
                        y_pred = pred_df["class"].values
                        if len(y_true) == len(y_pred):
                            precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                            print(f"Detection Metrics:")
                            print(f"Precision: {precision:.3f}")
                            print(f"Recall: {recall:.3f}")
                            f.write(f"Precision: {precision:.3f}\n")
                            f.write(f"Recall: {recall:.3f}\n")
                        else:
                            print("Mismatch in ground truth and predictions length. Skipping Precision/Recall.")
                            f.write("Precision: N/A (mismatch)\n")
                            f.write("Recall: N/A (mismatch)\n")

                        iou_threshold = 0.5
                        ap_sum = 0
                        num_classes = len(set(gt_df["class"]).union(set(pred_df["class"])))
                        for cls in self.valid_classes:
                            gt_cls = gt_df[gt_df["class"] == cls]
                            pred_cls = pred_df[pred_df["class"] == cls]
                            if gt_cls.empty or pred_cls.empty:
                                continue
                            ious = []
                            for _, pred in pred_cls.iterrows():
                                pred_box = [pred["x_min"], pred["y_min"], pred["x_max"], pred["y_max"]]
                                best_iou = 0
                                for _, gt in gt_cls.iterrows():
                                    gt_box = [gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]]
                                    iou = self.compute_iou(pred_box, gt_box)
                                    best_iou = max(best_iou, iou)
                                ious.append(best_iou >= iou_threshold)
                            if ious:
                                precision_vals = []
                                recall_vals = []
                                true_positives = sum(ious)
                                false_positives = len(ious) - true_positives
                                false_negatives = len(gt_cls) - true_positives
                                for i in range(1, len(ious) + 1):
                                    prec = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                                    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                                    precision_vals.append(prec)
                                    recall_vals.append(rec)
                                ap = np.trapz(precision_vals, recall_vals) if precision_vals and recall_vals else 0
                                ap_sum += ap
                        map50 = ap_sum / num_classes if num_classes > 0 else 0
                        print(f"mAP@0.5: {map50:.3f}")
                        f.write(f"mAP@0.5: {map50:.3f}\n")
                else:
                    print("No ground truth provided. Skipping detection metrics.")
                    f.write("Precision: N/A (no ground truth)\n")
                    f.write("Recall: N/A (no ground truth)\n")
                    f.write("mAP@0.5: N/A (no ground truth)\n")
            except Exception as e:
                print(f"Detection metrics computation failed: {e}")
                f.write(f"Detection Metrics: Failed to compute: {e}\n")

            try:
                acc = mm.MOTAccumulator(auto_id=True)
                if os.path.exists(gt_file):
                    gt_df = pd.read_csv(gt_file)
                    for frame_id, frame_preds in enumerate(predictions):
                        gt_frame = gt_df[gt_df["timestamp_ms"] == frame_id * (1000 / 30)]
                        gt = np.array([[i, row["x_min"], row["y_min"], row["x_max"] - row["x_min"], row["y_max"] - row["y_min"]]
                                       for i, row in gt_frame.iterrows()]) if not gt_frame.empty else np.array([])
                        pred = np.array([[p['id'], *p['bbox']] for p in frame_preds]) if frame_preds else np.array([])
                        dists = mm.distances.iou_matrix(gt[:, 1:], pred[:, 1:] if pred.size else np.array([]), max_iou=0.5)
                        acc.update(gt[:, 0] if gt.size else [], pred[:, 0] if pred.size else [], dists)
                else:
                    print("No ground truth for tracking metrics.")
                    f.write("Tracking Metrics: Skipped (ground truth missing)\n")
                    return

                mh = mm.metrics.create()
                summary = mh.compute(acc, metrics=['mota', 'motp', 'idsw'], name='acc')
                mota = summary['mota']['acc']
                motp = summary['motp']['acc']
                idsw = summary['idsw'].iloc[0]
                print(f"\nTracking Metrics:")
                print(f"MOTA: {mota:.3f}")
                print(f"MOTP: {motp:.3f}")
                print(f"ID Switches: {idsw}")
                f.write("\nTracking Metrics:\n")
                f.write(f"MOTA: {mota:.3f}\n")
                f.write(f"MOTP: {motp:.3f}\n")
                f.write(f"ID Switches: {idsw}\n")
            except Exception as e:
                print(f"Tracking metrics computation failed: {e}")
                f.write(f"Tracking Metrics: Failed to compute: {e}\n")

        print(f"Metrics saved to: {metrics_file}")

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_g, y1_g, x2_g, y2_g = box2
        xi1 = max(x1, x1_g)
        yi1 = max(y1, y1_g)
        xi2 = min(x2, x2_g)
        yi2 = min(y2, y2_g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_g - x1_g) * (y2_g - y1_g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

def parse_args():
    parser = argparse.ArgumentParser(description="Détection des objets avec YOLOv11")
    parser.add_argument("--filepath", type=str, required=True, help="Chemin de la vidéo")
    parser.add_argument("--targets", type=str, nargs='+', default=["all"], help="Objets à détecter")
    parser.add_argument("--confidence", type=float, default=0.01, help="Seuil de confiance")
    parser.add_argument("--no-window", action="store_true", help="Désactiver la fenêtre OpenCV", default=True)
    parser.add_argument("--evaluate", action="store_true", help="Calculer les métriques")
    return parser.parse_args()

def main():
    args = parse_args()
    detector = Detector(args.filepath, args.targets, no_window=args.no_window, evaluate=args.evaluate)
    detector.process_video(confidence=args.confidence)

if __name__ == "__main__":
    main()