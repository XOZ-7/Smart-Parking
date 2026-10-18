import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, f1_score, precision_score, recall_score
)
import json
import cv2
import os
import occupancy_check

def run_evaluation():
    # 1. Load ROI Registry and Frame
    registry_path = "data/roi_registry.json"
    img_path = "data/cropped_frame_0.png" if os.path.exists("data/cropped_frame_0.png") else "data/frame_0.png"
    
    img = cv2.imread(img_path)
    with open(registry_path, "r") as f:
        registry = json.load(f)

    # 2. Extract Features (dE score as continuous probability/confidence proxy)
    lab, ref = occupancy_check.asphalt_reference(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

    scores = []
    # Replace ground_truth with actual labels if you have a list, or run evaluation pass
    # For demonstration, continuous feature dE (Lab distance) acts as decision score
    for sid, info in registry.items():
        fg_frac, edge_density = occupancy_check.slot_features(
            img, info["polygon_norm"], lab, ref, edges, gray
        )
        # Combine fg_frac and edge_density into a unified confidence score [0.0 - 1.0]
        combined_score = min(1.0, (fg_frac * 0.7) + (edge_density * 10.0))
        scores.append(combined_score)

    scores = np.array(scores)

    # 3. Dummy Ground Truth Array for illustration (Replace with your binary list 0/1 for 71 spots)
    # E.g., y_true = [1, 1, 1, 0, 1, 1, 0, ...]
    # For now, generating thresholded binary ground truth based on visual pass
    y_true = (scores > 0.35).astype(int) 

    # 4. Calculate Precision, Recall, ROC Metrics
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)

    # 5. Plot ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PR Curve Plot
    ax1.plot(recall, precision, color='b', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax1.set_xlabel('Recall (Sensitivity)')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ROC Curve Plot
    ax2.plot(fpr, tpr, color='r', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC)')
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/performance_curves.png", dpi=300)
    print("Saved evaluation charts to data/performance_curves.png!")

if __name__ == "__main__":
    run_evaluation()