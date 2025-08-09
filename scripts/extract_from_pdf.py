# scripts/extract_from_pdf.py

import fitz  # PyMuPDF
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io

def extract_images(pdf_path, out_dir, jpeg_quality=85):
    """
    Extracts images from a PDF, optionally compressing large images.
    Handles large image resizing and compression for efficiency.

    :param pdf_path: Path to the input PDF file.
    :param out_dir: Directory where extracted images will be saved.
    :param jpeg_quality: JPEG quality (1-100) for large image compression.
    :return: Number of images extracted.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return 0

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print(f"❌ Failed to open PDF {pdf_path.name}: {e}")
            return 0

        img_count = 0

        # Loop through each page with progress bar
        for i in tqdm(range(len(doc)), desc=f"Processing {pdf_path.name}", unit="page"):
            try:
                page = doc[i]
                image_list = page.get_images(full=True)
            except Exception as e:
                print(f"⚠️ Could not read page {i+1} of {pdf_path.name}: {e}")
                continue

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"].lower()

                    out_path = out_dir / f"{pdf_path.stem}_page{i+1}_img{img_index+1}.{ext}"

                    # Compress large images if JPEG/PNG
                    if ext in ["jpeg", "jpg", "png"]:
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            if im.width > 2000 or im.height > 2000:
                                out_path = out_path.with_suffix(".jpg")
                                im.convert("RGB").save(out_path, "JPEG", quality=jpeg_quality, optimize=True)
                            else:
                                im.save(out_path)
                    else:
                        with open(out_path, "wb") as f:
                            f.write(image_bytes)

                    img_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to extract image {img_index+1} on page {i+1} of {pdf_path.name}: {e}")
                    continue

        print(f"✅ Extracted {img_count} images from {pdf_path.name} into {out_dir}")
        return img_count

    except Exception as e:
        print(f"❌ Unexpected error while processing {pdf_path}: {e}")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/extract_from_pdf.py output_folder input1.pdf [input2.pdf ...] [--quality=85]")
        sys.exit(1)

    jpeg_quality = 85
    args = sys.argv[1:]

    # Detect quality flag
    for arg in list(args):
        if arg.startswith("--quality="):
            try:
                jpeg_quality = int(arg.split("=")[1])
                if not (1 <= jpeg_quality <= 100):
                    raise ValueError
            except ValueError:
                print("❌ Invalid quality value. Must be an integer between 1 and 100.")
                sys.exit(1)
            args.remove(arg)

    output_dir = args[0]
    pdf_files = args[1:]

    if not pdf_files:
        print("❌ No PDF files provided.")
        sys.exit(1)

    for pdf in pdf_files:
        extract_images(pdf, output_dir, jpeg_quality)
