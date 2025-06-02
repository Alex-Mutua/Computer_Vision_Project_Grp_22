# heatmap_generator.py
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_heatmap():
    """Generate class-specific heatmaps from position_log.csv."""
    try:
        base_dir = "Documents/projet-aims"
        input_path = os.path.join(base_dir, "src", "outputs", "position_log.csv")
        output_dir = os.path.join(base_dir, "src", "outputs")

        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            return False

        # Group positions by class
        positions_by_class = {}
        with open(input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x_center = (int(row["x_min"]) + int(row["x_max"])) / 2
                    y_center = (int(row["y_min"]) + int(row["y_max"])) / 2
                    class_name = row["class"].lower()
                    if class_name not in positions_by_class:
                        positions_by_class[class_name] = []
                    positions_by_class[class_name].append([x_center, y_center])
                except (KeyError, ValueError) as e:
                    print(f"Error reading row {row}: {e}")
                    continue

        if not positions_by_class:
            print("No valid positions found in position_log.csv")
            return False

        # Generate heatmap per class
        for class_name, positions in positions_by_class.items():
            positions = np.array(positions)
            heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=50)

            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap.T, cmap="YlOrRd", cbar=True)
            plt.title(f"Carte thermique: {class_name.capitalize()}")
            plt.xlabel("X")
            plt.ylabel("Y")
            output_path = os.path.join(output_dir, f"heatmap_{class_name.replace(' ', '_')}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Heatmap generated for {class_name}: {output_path}")

        return True
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return False

if __name__ == "__main__":
    generate_heatmap()