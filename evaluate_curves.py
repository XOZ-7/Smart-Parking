import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import occupancy_check

def generate_publication_curves():
    img_path = "data/cropped_frame_0.png" if os.path.exists("data/cropped_frame_0.png") else "data/frame_0.png"
    reg_path = "data/roi_registry.json"
    gt_path = "data/ground_truth.json"

    if not os.path.exists(gt_path):
        import create_gt

    img = cv2.imread(img_path)
    with open(reg_path, "r") as f:
        registry = json.load(f)
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    lab, ref = occupancy_check.asphalt_reference(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

    # Collect ground truth vector and raw continuous features
    y_true = []
    raw_fg_frac = []
    raw_edge_density = []

    for sid, info in registry.items():
        if sid in ground_truth:
            y_true.append(ground_truth[sid])
            fg_frac, edge_density = occupancy_check.slot_features(
                img, info["polygon_norm"], lab, ref, edges, gray
            )
            raw_fg_frac.append(fg_frac)
            raw_edge_density.append(edge_density)

    y_true = np.array(y_true)
    raw_fg_frac = np.array(raw_fg_frac)
    raw_edge_density = np.array(raw_edge_density)

    # Parameter sweep across 6 threshold sensitivity profiles (diff=20 to diff=45)
    diff_thresholds = [20, 25, 30, 35, 40, 45]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, (ax_roc, ax_pr) = plt.subplots(2, 1, figsize=(7, 10))

    for idx, diff_val in enumerate(diff_thresholds):
        # Calculate continuous decision probability map scaled by sensitivity parameter
        scores = np.clip((raw_fg_frac * (diff_val / 25.0)) + (raw_edge_density * 8.0), 0.0, 1.0)

        # Sweep 100 internal decision thresholds for curve smoothing
        sweep_th = np.linspace(0.0, 1.0, 100)
        tprs, fprs, precisions, recalls = [], [], [], []

        for th in sweep_th:
            preds = (scores >= th).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            tn = np.sum((y_true == 0) & (preds == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tpr

            tprs.append(tpr)
            fprs.append(fpr)
            precisions.append(prec)
            recalls.append(rec)

        tprs, fprs = np.array(tprs), np.array(fprs)
        precisions, recalls = np.array(precisions), np.array(recalls)

        # Calculate AUC area
        roc_auc = np.abs(np.trapz(tprs, fprs))
        pr_auc = np.abs(np.trapz(precisions, recalls))

        # Plot ROC curve
        ax_roc.plot(fprs, tprs, '--o', color=colors[idx], markersize=3, lw=1.5,
                    label=f'{idx}: diff={diff_val} AUC={roc_auc:.3f}')

        # Plot PR curve
        ax_pr.plot(recalls, precisions, '--o', color=colors[idx], markersize=3, lw=1.5,
                   label=f'{idx}: diff={diff_val} AP={pr_auc:.3f}')

    # ROC Formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_title("ROC Curve", fontweight='bold', fontsize=12)
    ax_roc.set_xlabel("False positive rate (1 - Specificity)", fontsize=10)
    ax_roc.set_ylabel("True positive rate (Recall)", fontsize=10)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle=':', alpha=0.6)

    # PR Formatting
    ax_pr.set_title("PR Curve", fontweight='bold', fontsize=12)
    ax_pr.set_xlabel("True positive rate (Recall)", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/research_publication_curves.png", dpi=300)
    print("Saved publication curves to data/research_publication_curves.png!")

if __name__ == "__main__":
    generate_publication_curves()