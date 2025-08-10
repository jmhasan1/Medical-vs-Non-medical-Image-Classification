import os
import time
import torch
import pandas as pd
from PIL import Image
from scripts.cluster_images import get_basic_transforms, load_mobilenet_model, extract_mobilenet_embedding
from scripts.clip_cluster import load_clip_model, extract_clip_embedding
import joblib

def infer(images_folder, kmeans_mobilenet_path, kmeans_clip_path, device='cuda'):
    mobilenet_model = load_mobilenet_model(device)
    clip_model, preprocess = load_clip_model(device)
    kmeans_mobilenet = joblib.load(kmeans_mobilenet_path)
    kmeans_clip = joblib.load(kmeans_clip_path)

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    times_mobilenet = []
    times_clip = []
    preds_mobilenet = []
    preds_clip = []

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = Image.open(img_path).convert('RGB')

        # MobileNet inference
        start = time.perf_counter()
        emb_mn = extract_mobilenet_embedding(mobilenet_model, img, device)
        pred_mn = kmeans_mobilenet.predict([emb_mn])[0]
        end = time.perf_counter()
        times_mobilenet.append(end - start)
        preds_mobilenet.append((img_file, pred_mn))

        # CLIP inference
        start = time.perf_counter()
        emb_clip = extract_clip_embedding(clip_model, preprocess, img, device)
        pred_clip = kmeans_clip.predict([emb_clip])[0]
        end = time.perf_counter()
        times_clip.append(end - start)
        preds_clip.append((img_file, pred_clip))

    avg_time_mn = sum(times_mobilenet) / len(times_mobilenet)
    avg_time_clip = sum(times_clip) / len(times_clip)

    print(f"MobileNet + KMeans average inference time per image: {avg_time_mn:.4f} seconds")
    print(f"CLIP + KMeans average inference time per image: {avg_time_clip:.4f} seconds")

    # Save predictions CSV
    pd.DataFrame(preds_mobilenet, columns=['image', 'cluster']).to_csv('mobilenet_predictions.csv', index=False)
    pd.DataFrame(preds_clip, columns=['image', 'cluster']).to_csv('clip_predictions.csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inference script with timing')
    parser.add_argument('--images', type=str, required=True, help='Folder of images to infer')
    parser.add_argument('--kmeans_mobilenet', type=str, required=True, help='Path to MobileNet KMeans model (pickle)')
    parser.add_argument('--kmeans_clip', type=str, required=True, help='Path to CLIP KMeans model (pickle)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    args = parser.parse_args()

    infer(args.images, args.kmeans_mobilenet, args.kmeans_clip, args.device)
