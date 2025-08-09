import os
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

def preprocess_images(source_dirs, output_dir, image_size=(224, 224), min_size=200):
    """
    Cleans and preprocesses images from multiple source directories.
    - Removes corrupted or very small images
    - Deduplicates using perceptual hash
    - Resizes to standard size
    - Saves in processed output directory (unlabeled for now)

    :param source_dirs: List of source directories containing raw images
    :param output_dir: Directory to save cleaned & resized images
    :param image_size: Desired output size (width, height)
    :param min_size: Minimum width/height for keeping an image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    cleaned_count = 0

    for src_dir in source_dirs:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            print(f"⚠️ Source folder not found: {src_dir}")
            continue

        for img_file in tqdm(list(src_dir.glob("*.*")), desc=f"Processing {src_dir.name}"):
            try:
                with Image.open(img_file) as img:
                    img = img.convert("RGB")

                    # Skip small images
                    if img.width < min_size or img.height < min_size:
                        continue

                    # Deduplicate
                    img_hash = imagehash.average_hash(img)
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    # Resize
                    img = img.resize(image_size)

                    # Save cleaned image
                    out_path = output_dir / f"{src_dir.stem}_{img_file.name}"
                    img.save(out_path, format="JPEG", quality=95)
                    cleaned_count += 1

            except Exception as e:
                print(f"⚠️ Skipping {img_file} due to error: {e}")
                continue

    print(f"✅ Preprocessed {cleaned_count} images into {output_dir}")


if __name__ == "__main__":
    # Input: Raw images from PDFs and URLs
    source_folders = [
        "data/img_pdf_extracted",
        "data/img_url_extracted"
    ]

    # Output: All cleaned images in one folder
    preprocess_images(source_folders, "data/processed")

