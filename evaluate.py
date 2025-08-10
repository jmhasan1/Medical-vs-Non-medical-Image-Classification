# evaluate.py
import csv, time, argparse, os
from infer import run_inference_folder

def evaluate(folder, labels_csv, cnn_weights=None):
    # labels_csv : CSV with columns 'path,label' where label is 'medical' or 'non-medical'
    gt = {}
    with open(labels_csv, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if len(row) >= 2:
                gt[row[0]] = row[1]
    preds = run_inference_folder(folder, cnn_weights=cnn_weights)
    total = 0
    correct_clip = 0
    correct_cnn = 0
    for r in preds:
        img_rel = os.path.basename(r['image'])
        if img_rel not in gt:
            continue
        total += 1
        if r['clip'] == gt[img_rel]:
            correct_clip += 1
        if r['cnn'] is not None and r['cnn'][0] == gt[img_rel]:
            correct_cnn += 1
    return {
        "total": total,
        "clip_acc": correct_clip/total if total else None,
        "cnn_acc": correct_cnn/total if total else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--cnn_weights", default=None)
    args = parser.parse_args()
    res = evaluate(args.folder, args.labels_csv, args.cnn_weights)
    print(res)
