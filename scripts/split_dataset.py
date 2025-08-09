# scripts/split_dataset.py
import argparse
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(processed_folder, labels_csv, out_root, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    df = pd.read_csv(labels_csv)
    src = Path(processed_folder)
    out_root = Path(out_root)
    for split in ["train","val","test"]:
        for cls in ["medical","non-medical"]:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    # stratified split
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df['label'], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_frac/(train_frac+val_frac), stratify=train_val['label'], random_state=seed)

    def copy_rows(rows, split_name):
        for _, r in rows.iterrows():
            src_path = src / r['filename']
            dst = out_root / split_name / r['label'] / r['filename']
            if src_path.exists():
                shutil.copy2(src_path, dst)
    copy_rows(train, "train")
    copy_rows(val, "val")
    copy_rows(test, "test")
    print("Dataset splitted into", out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="sample_data")
    args = parser.parse_args()
    split_dataset(args.processed, args.labels, args.out)
