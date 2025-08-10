import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.models as models
from sklearn.cluster import KMeans
import numpy as np
import shutil
import joblib
import os

from utils.preprocess import get_basic_transforms


def extract_features(image_paths, model, transform, device):
    """
    Extract embeddings from a list of images using a pretrained CNN.
    """
    features = []
    valid_paths = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                feat = model(tensor).cpu().numpy().flatten()
                features.append(feat)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

    return np.array(features), valid_paths


def cluster_images(data_dir, out_csv, n_clusters=2, img_size=224, output_folders=True, rename_map=None):
    data_dir = Path(data_dir)
    image_paths = list(data_dir.glob("**/*.*"))

    if not image_paths:
        print("❌ No images found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MobileNetV2 as a feature extractor
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()  # Remove classification layer
    model = model.to(device)
    model.eval()

    transform = get_basic_transforms(img_size)

    # Extract embeddings
    features, valid_paths = extract_features(image_paths, model, transform, device)

    # KMeans clustering
    print("🔍 Running KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    preds = kmeans.fit_predict(features)

     # --- New code: Save the trained KMeans model ---
    os.makedirs('models', exist_ok=True)  # Create models directory if it doesn't exist
    joblib.dump(kmeans, 'models/kmeans_mobilenet.pkl')  # Save model as pickle file
    print("Saved MobileNet KMeans model to models/kmeans_mobilenet.pkl")

    # Save predictions CSV
    df = pd.DataFrame({
        "image_path": [str(p) for p in valid_paths],
        "cluster": preds
    })
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved predictions to {out_csv}")

    # Optional: Create separate folders for clusters
    if output_folders:
        out_dir = Path(out_csv).parent
        for cluster_id in range(n_clusters):
            folder_name = f"cluster_{cluster_id}"
            if rename_map and cluster_id in rename_map:
                folder_name = rename_map[cluster_id]
            cluster_folder = out_dir / folder_name
            cluster_folder.mkdir(parents=True, exist_ok=True)

            cluster_files = df[df["cluster"] == cluster_id]["image_path"].tolist()
            for file_path in cluster_files:
                try:
                    shutil.copy(file_path, cluster_folder / Path(file_path).name)
                except Exception as e:
                    print(f"⚠️ Failed to copy {file_path}: {e}")

        print(f"📂 Cluster folders created in: {out_dir}")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster images into groups using MobileNetV2 + KMeans")
    parser.add_argument("--src", required=True, help="Directory with images")
    parser.add_argument("--out", required=True, help="Output CSV file with cluster predictions")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters")
    parser.add_argument("--size", type=int, default=224, help="Image size for preprocessing")
    parser.add_argument("--no-folders", action="store_true", help="Disable creating cluster folders")
    parser.add_argument("--rename", nargs="+", help="Rename clusters e.g. 0=medical 1=non-medical")
    args = parser.parse_args()

    # Parse rename mapping if provided
    rename_map = None
    if args.rename:
        rename_map = {}
        for mapping in args.rename:
            try:
                cluster_id, name = mapping.split("=")
                rename_map[int(cluster_id)] = name
            except ValueError:
                print(f"⚠️ Skipping invalid rename mapping: {mapping}")

    cluster_images(
        args.src,
        args.out,
        args.clusters,
        args.size,
        output_folders=not args.no_folders,
        rename_map=rename_map
    )

