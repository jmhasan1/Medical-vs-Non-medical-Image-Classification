import sys
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
import io
from urllib.parse import urljoin

def extract_images_from_url(url, out_dir, min_size=200, allowed_ext=("jpg", "jpeg", "png")):
    """
    Downloads and saves images from a webpage, with filters for size and extension.
    
    :param url: Webpage URL
    :param out_dir: Directory to save extracted images
    :param min_size: Minimum width/height for image
    :param allowed_ext: Tuple of allowed file extensions
    :return: Number of images downloaded
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to retrieve {url}: {e}")
            return 0

        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")
        if not img_tags:
            print(f"⚠️ No images found on {url}")
            return 0

        img_count = 0

        for idx, img_tag in enumerate(tqdm(img_tags, desc=f"Processing {url}", unit="img")):
            img_url = img_tag.get("src")
            if not img_url:
                continue

            # Handle relative URLs
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("/"):
                img_url = urljoin(url, img_url)

            # Filter by extension
            ext = img_url.split(".")[-1].lower().split("?")[0]
            if ext not in allowed_ext:
                continue

            try:
                img_data = requests.get(img_url, timeout=10).content
                im = Image.open(io.BytesIO(img_data))

                # Skip small images
                if im.width < min_size or im.height < min_size:
                    continue

                out_path = out_dir / f"{Path(url).stem}_img{idx+1}.{ext}"
                im.save(out_path)
                img_count += 1
            except Exception as e:
                print(f"⚠️ Could not save image {img_url} from {url}: {e}")
                continue

        print(f"✅ Downloaded {img_count} images from {url} into {out_dir}")
        return img_count

    except Exception as e:
        print(f"❌ Unexpected error while processing {url}: {e}")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/extract_from_url.py output_folder url1 [url2 ...]")
        sys.exit(1)

    output_dir = sys.argv[1]
    urls = sys.argv[2:]

    if not urls:
        print("❌ No URLs provided.")
        sys.exit(1)

    for url in urls:
        extract_images_from_url(url, output_dir)
