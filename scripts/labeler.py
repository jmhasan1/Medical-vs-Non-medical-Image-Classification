# scripts/labeler.py
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import csv

def label_images(src_folder, out_csv):
    src = Path(src_folder)
    files = sorted([p for p in src.glob("*.*")])
    if not files:
        print("No image files found in", src)
        return
    out_csv = Path(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        # If file empty, write header
        if f.tell()==0:
            writer.writerow(["filename","label"])
        for p in files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print("Skip unreadable", p.name, e)
                continue
            plt.imshow(img)
            plt.title(p.name)
            plt.axis("off")
            plt.show(block=False)
            label = input("Label [m]=medical, [n]=non-medical, [s]=skip, [q]=quit: ").strip().lower()
            plt.close()
            if label == 'q':
                print("Exiting.")
                return
            if label == 's' or label == '':
                continue
            if label == 'm':
                writer.writerow([p.name, "medical"])
            elif label == 'n':
                writer.writerow([p.name, "non-medical"])
            else:
                print("Unknown label, skipping.")
    print("Saved labels to", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source folder to label (e.g. data/processed)")
    parser.add_argument("--out", default="data/labels.csv", help="output CSV file")
    args = parser.parse_args()
    label_images(args.src, args.out)
