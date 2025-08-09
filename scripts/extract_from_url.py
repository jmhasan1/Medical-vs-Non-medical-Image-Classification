import sys
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
import io

def extract_images_from_url(url, out_dir):
    """
    Downloads and saves images from a webpage.
    :param url: Webpage URL
    :param out_dir: Directory to save extracted images
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
                from urllib.parse import urljoin
                img_url = urljoin(url, img_url)

            try:
                img_data = requests.get(img_url, timeout=10).content
                im = Image.open(io.BytesIO(img_data))

                # Determine file extension
                ext = im.format.lower() if im.format else "jpg"
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
