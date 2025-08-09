# scripts/clip_cluster.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import sys
import torch
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
from utils.preprocess import get_basic_transforms
import shutil

# ---- Auto batch size detection ----
def get_optimal_batch_size(default=32):
    if not torch.cuda.is_available():
        return default
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
    if total_mem < 4096:
        return 4
    elif total_mem < 6144:
        return 8
    elif total_mem < 8192:
        return 16
    else:
        return default

# ---- Zero-shot label assignment ----
def zero_shot_label_cluster(centroid, model, preprocess, device):
    # Rich, context-based prompts
    prompts = [
        "A diagnostic medical image such as an X-ray, MRI scan, CT scan, or ultrasound from a hospital",
        "A non-medical photograph such as landscapes, architecture, nature, animals, or everyday objects"
    ]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        centroid_tensor = torch.tensor(centroid).unsqueeze(0).to(device)
        centroid_tensor /= centroid_tensor.norm(dim=-1, keepdim=True)
        centroid_tensor = centroid_tensor.to(text_features.dtype)
        sims = (centroid_tensor @ text_features.T).squeeze(0).cpu().numpy()

    return np.argmax(sims)  # 0 = medical, 1 = non-medical

def save_images_by_label(df, save_dir):
    save_dir = Path(save_dir)
    (save_dir / "medical").mkdir(parents=True, exist_ok=True)
    (save_dir / "non_medical").mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        src_path = Path(row["path"])
        if row["predicted_label"] == "medical":
            shutil.copy(src_path, save_dir / "medical" / src_path.name)
        else:
            shutil.copy(src_path, save_dir / "non_medical" / src_path.name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder with processed images")
    parser.add_argument("--out", required=True, help="Output CSV file")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for embedding extraction")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model variant")
    parser.add_argument("--save_dir", default=None, help="Optional directory to save classified images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or get_optimal_batch_size()

    print(f"🚀 Using device: {device}, batch size: {batch_size}")

    # Load CLIP
    model, preprocess_clip = clip.load(args.model, device=device)
    transform = get_basic_transforms()

    # Load images
    src_dir = Path(args.src)
    image_paths = list(src_dir.glob("*.*"))
    if not image_paths:
        print(f"❌ No images found in {src_dir}")
        sys.exit(1)

    embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                img_t = preprocess_clip(img).unsqueeze(0)
                batch_images.append(img_t)
                valid_paths.append(str(p))
            except Exception as e:
                print(f"⚠️ Skipping {p} due to error: {e}")
        if not batch_images:
            continue
        batch_tensor = torch.cat(batch_images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Clustering
    kmeans = KMeans(n_clusters=args.clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)

    # Zero-shot mapping
    cluster_labels = {}
    for cluster_id in range(args.clusters):
        centroid = kmeans.cluster_centers_[cluster_id]
        assigned_label_idx = zero_shot_label_cluster(centroid, model, preprocess_clip, device)
        cluster_labels[cluster_id] = "medical" if assigned_label_idx == 0 else "non-medical"

    # Fallback if all same label
    assigned_counts = list(cluster_labels.values())
    if len(set(assigned_counts)) == 1:
        print("⚠️ Zero-shot mapping assigned same label to all clusters. Falling back to manual mapping.")
        # Assign first cluster as medical, second as non-medical
        cluster_labels = {0: "medical", 1: "non-medical"}

    df = pd.DataFrame({
        "path": valid_paths,
        "cluster_id": cluster_ids,
        "predicted_label": [cluster_labels[cid] for cid in cluster_ids]
    })

    df.to_csv(args.out, index=False)
    print(f"✅ Predictions saved to {args.out}")

    # Save classified images
    if args.save_dir:
        save_images_by_label(df, args.save_dir)
        print(f"📂 Classified images saved to {args.save_dir}")
