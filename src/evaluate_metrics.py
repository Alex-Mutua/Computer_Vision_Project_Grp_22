import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_detections(ground_truth_file, predictions_file, output_file, iou_threshold=0.5, time_threshold=100):
    """Evaluate detections against ground truth."""
    try:
        # Read ground truth
        gt = pd.read_csv(ground_truth_file)
        gt = gt[["timestamp_ms", "class", "x_min", "y_min", "x_max", "y_max"]]

        # Read predictions
        pred = pd.read_csv(predictions_file)
        pred = pred[["timestamp_ms", "track_id", "class", "x_min", "y_min", "x_max", "y_max"]]

        # Match detections
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt)
        id_switches = 0
        previous_ids = {}

        for _, gt_row in gt.iterrows():
            gt_time = gt_row["timestamp_ms"]
            gt_box = [gt_row["x_min"], gt_row["y_min"], gt_row["x_max"], gt_row["y_max"]]
            gt_class = gt_row["class"].lower()

            # Find predictions within time threshold
            matches = pred[
                (abs(pred["timestamp_ms"] - gt_time) <= time_threshold) &
                (pred["class"].str.lower() == gt_class)
            ]

            best_iou = 0
            best_pred = None
            for _, pred_row in matches.iterrows():
                pred_box = [pred_row["x_min"], pred_row["y_min"], pred_row["x_max"], pred_row["y_max"]]
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_row

            if best_iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
                # Check ID consistency
                pred_id = best_pred["track_id"]
                if gt_class in previous_ids and previous_ids[gt_class] != pred_id:
                    id_switches += 1
                previous_ids[gt_class] = pred_id
            else:
                false_positives += len(matches)

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Write report
        with open(output_file, "w") as f:
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
            f.write(f"ID Switches: {id_switches}\n")
            f.write(f"True Positives: {true_positives}\n")
            f.write(f"False Positives: {false_positives}\n")
            f.write(f"False Negatives: {false_negatives}\n")

        print(f"Evaluation completed. Report saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return False

if __name__ == "__main__":
    evaluate_detections(
        ground_truth_file="../outputs/ground_truth_template.csv",
        predictions_file="../outputs/position_log.csv",
        output_file="../outputs/evaluation_report.txt",
        iou_threshold=0.5,
        time_threshold=100
    )