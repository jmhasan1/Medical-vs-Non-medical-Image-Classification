import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from tqdm import tqdm


def analyze_images(data_dir):
    """
    Basic stats: count, formats, size distribution
    """
    data_dir = Path(data_dir)
    image_files = list(data_dir.glob("**/*.*"))

    if not image_files:
        print("❌ No images found in directory.")
        return

    formats = []
    widths = []
    heights = []

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                formats.append(img.format)
                widths.append(img.width)
                heights.append(img.height)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    print(f"📊 Total images: {len(image_files)}")
    print(f"📁 Unique formats: {set(formats)}")
    print(f"📏 Width range: {min(widths)}–{max(widths)}")
    print(f"📏 Height range: {min(heights)}–{max(heights)}")

    # Format distribution
    plt.figure(figsize=(6, 4))
    format_counts = Counter(formats)
    plt.bar(format_counts.keys(), format_counts.values())
    plt.title("Image Formats Distribution")
    plt.ylabel("Count")
    plt.show()

    # Size distribution
    plt.figure(figsize=(6, 4))
    plt.hist(widths, bins=20, alpha=0.7, label="Width")
    plt.hist(heights, bins=20, alpha=0.7, label="Height")
    plt.legend()
    plt.title("Image Dimensions Distribution")
    plt.show()


def visualize_with_pca(data_dir, sample_size=100):
    """
    Extract MobileNetV2 features & visualize with PCA (2D)
    """
    data_dir = Path(data_dir)
    image_files = list(data_dir.glob("**/*.*"))[:sample_size]

    if not image_files:
        print("❌ No images found for PCA.")
        return

    # Load pretrained MobileNetV2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()  # remove classification head
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    features = []

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Extracting features"):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                feat = model(tensor).cpu().numpy().flatten()
                features.append(feat)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")

    features = np.array(features)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title("PCA of Image Features (Unsupervised View)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for Unsupervised Image Dataset")
    parser.add_argument("--src", required=True, help="Directory with images")
    parser.add_argument("--pca", action="store_true", help="Run PCA visualization")
    parser.add_argument("--sample", type=int, default=100, help="Number of images for PCA")
    args = parser.parse_args()

    analyze_images(args.src)

    if args.pca:
        visualize_with_pca(args.src, args.sample)
