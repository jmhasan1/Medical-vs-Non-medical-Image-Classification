import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import argparse

def best_map(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))
    new_preds = np.array([mapping.get(c, c) for c in pred_labels])
    return new_preds

def evaluate(pred_csv, gt_csv, image_col='image', pred_col='cluster', label_col='label'):
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    merged = pd.merge(pred_df, gt_df, on=image_col, how='inner')
    true_labels = merged[label_col].values
    pred_labels = merged[pred_col].values

    mapped_preds = best_map(true_labels, pred_labels)
    acc = accuracy_score(true_labels, mapped_preds)

    print(f"Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering predictions.')
    parser.add_argument('--predictions', type=str, required=True, help='CSV file with predictions (image, cluster)')
    parser.add_argument('--groundtruth', type=str, required=True, help='CSV file with ground truth labels (image, label)')
    args = parser.parse_args()

    evaluate(args.predictions, args.groundtruth)
