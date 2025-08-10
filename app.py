# app.py
"""
Gradio app for Medical vs Non-Medical Image Classification
- Supports: Upload PDF, enter a webpage URL, or upload images
- Uses: open_clip (zero-shot) + optional MobileNetV2 fine-tuned CNN (best_model.pth)
- Designed for Hugging Face Spaces (Gradio). Keep models small for resource limits.
"""

import os
import io
import time
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
import requests
from bs4 import BeautifulSoup

import gradio as gr

# open_clip (installed via requirements)
import torch
try:
    import open_clip
except Exception as e:
    open_clip = None
    print("open_clip not available:", e)

# torchvision for optional CNN
import torchvision.transforms as T
import torchvision.models as models

# PyMuPDF for PDF extraction
try:
    import fitz
except Exception:
    fitz = None

# -------------------------
# Configuration / Constants
# -------------------------
TEXT_LABELS = ["medical", "non-medical"]
CLIP_MODEL_NAME = "ViT-B-32"   # relatively small/fast
CLIP_PRETRAINED = "openai"
CNN_WEIGHTS = "best_model.pth"  # optional: place in repo root to enable CNN inference
IMG_SIZE = 224                 # CLIP preprocess will handle cropping; CNN uses resize
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utility: Image Extraction
# -------------------------
def extract_images_from_pdf_bytes(pdf_bytes: bytes, out_dir: str, max_images: int = 200) -> List[str]:
    """
    Extract images from a PDF provided as bytes using PyMuPDF (fitz).
    Returns list of saved image paths.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction. Install pymupdf.")
    saved = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        page = doc[i]
        imglist = page.get_images(full=True)
        for img_index, img in enumerate(imglist):
            if len(saved) >= max_images:
                break
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            out_path = Path(out_dir) / f"page{i+1}_img{img_index+1}.{ext}"
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            saved.append(str(out_path))
    return saved

def extract_images_from_url(url: str, out_dir: str, max_images: int = 200) -> List[str]:
    """
    Download <img> sources from a web page.
    """
    saved = []
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
    except Exception as e:
        print("failed fetching url:", e)
        return saved
    soup = BeautifulSoup(r.text, "html.parser")
    imgs = soup.find_all("img")
    for i, img in enumerate(imgs):
        if len(saved) >= max_images:
            break
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue
        img_url = requests.compat.urljoin(url, src)
        try:
            res = requests.get(img_url, timeout=10)
            if res.status_code == 200 and "image" in res.headers.get("Content-Type", ""):
                ext = res.headers.get("Content-Type").split("/")[-1].split(";")[0]
                fname = f"web_img_{i+1}.{ext}"
                out_path = Path(out_dir) / fname
                with open(out_path, "wb") as f:
                    f.write(res.content)
                saved.append(str(out_path))
        except Exception as e:
            # skip problematic images
            continue
    return saved

def load_pil_image(path_or_bytes) -> Image.Image:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        return Image.open(path_or_bytes).convert("RGB")

# -------------------------
# CLIP (open_clip) loading
# -------------------------
_clip_model = None
_clip_preprocess = None

def load_clip(model_name=CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE):
    global _clip_model, _clip_preprocess
    if _clip_model is not None and _clip_preprocess is not None:
        return _clip_model, _clip_preprocess
    if open_clip is None:
        raise RuntimeError("open_clip not installed. Add git+https://github.com/mlfoundations/open_clip.git to requirements.")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    _clip_model = model
    _clip_preprocess = preprocess
    return _clip_model, _clip_preprocess

def clip_predict(image_pil: Image.Image, model, preprocess, device=DEVICE) -> Tuple[str, float]:
    img_t = preprocess(image_pil).unsqueeze(0).to(device)
    texts = TEXT_LABELS
    text_tokens = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.t()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    # return best label and confidence
    best_idx = int(probs.argmax())
    return texts[best_idx], float(probs[best_idx])

# -------------------------
# Optional CNN (MobileNetV2)
# -------------------------
_cnn_model = None
_cnn_transform = None
def load_cnn(weights_path=CNN_WEIGHTS, device=DEVICE):
    global _cnn_model, _cnn_transform
    if _cnn_model is not None:
        return _cnn_model, _cnn_transform
    # Load MobileNetV2
    model = models.mobilenet_v2(pretrained=False)
    # replace classifier
    num_ftrs = model.classifier[1].in_features
    import torch.nn as nn
    model.classifier[1] = nn.Linear(num_ftrs, len(TEXT_LABELS))
    # load weights if present
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt)
        print("Loaded CNN weights from", weights_path)
    else:
        # if not present, indicate unavailability
        print("CNN weights not found at", weights_path, "; CNN inference will be disabled.")
        return None, None
    model.to(device)
    model.eval()
    _cnn_model = model
    # transform for CNN
    _cnn_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return _cnn_model, _cnn_transform

def cnn_predict(image_pil: Image.Image, model, transform, device=DEVICE) -> Tuple[str, float]:
    x = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    # index mapping: 0 -> non-medical, 1 -> medical (as per training script)
    idx = int(probs.argmax())
    label = TEXT_LABELS[idx] if idx < len(TEXT_LABELS) else "unknown"
    return label, float(probs[idx])

# -------------------------
# Inference Pipeline
# -------------------------
def classify_images(image_paths: List[str], use_cnn: bool = True):
    """
    For each path, run CLIP and optionally CNN, return results list.
    """
    results = []
    # ensure CLIP loaded
    try:
        clip_model, clip_pre = load_clip()
    except Exception as e:
        return {"error": f"Failed to load CLIP: {e}\n{traceback.format_exc()}"}
    # load CNN if requested
    cnn_model, cnn_transform = None, None
    if use_cnn:
        try:
            cnn_model, cnn_transform = load_cnn()
            if cnn_model is None:
                use_cnn = False
        except Exception as e:
            print("Failed to load CNN:", e)
            use_cnn = False

    for p in image_paths:
        try:
            pil = load_pil_image(p)
        except Exception as e:
            continue
        # run CLIP
        try:
            clip_label, clip_conf = clip_predict(pil, clip_model, clip_pre)
        except Exception as e:
            clip_label, clip_conf = "error", 0.0
        cnn_label, cnn_conf = None, None
        if use_cnn and cnn_model is not None:
            try:
                cnn_label, cnn_conf = cnn_predict(pil, cnn_model, cnn_transform)
            except Exception as e:
                cnn_label, cnn_conf = None, None
        # ensemble rule: if both exist and agree -> choose that label; if disagree -> choose higher confidence
        final_label = clip_label
        final_conf = clip_conf
        if cnn_label is not None:
            if cnn_label == clip_label:
                # average confidence
                final_conf = (clip_conf + cnn_conf) / 2.0
                final_label = clip_label
            else:
                # choose higher confidence
                if cnn_conf > clip_conf:
                    final_label = cnn_label
                    final_conf = cnn_conf
        results.append({
            "image_path": p,
            "clip_label": clip_label,
            "clip_confidence": float(clip_conf),
            "cnn_label": cnn_label,
            "cnn_confidence": (float(cnn_conf) if cnn_conf is not None else None),
            "final_label": final_label,
            "final_confidence": float(final_conf)
        })
    return results

# -------------------------
# Gradio UI callbacks
# -------------------------
def handle_pdf_upload(pdf_file, use_cnn):
    if pdf_file is None:
        return "No file uploaded", None
    tmp = tempfile.TemporaryDirectory()
    try:
        pdf_bytes = pdf_file.read()
        img_paths = extract_images_from_pdf_bytes(pdf_bytes, tmp.name)
        if not img_paths:
            return "No images found in PDF.", None
        results = classify_images(img_paths, use_cnn=use_cnn)
        # prepare dataframe for download
        df = pd.DataFrame(results)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        # prepare images for display: pick first N thumbnails
        thumbs = []
        for r in results[:30]:
            pil = load_pil_image(r["image_path"])
            thumbnails_io = io.BytesIO()
            pil.resize((240,240)).save(thumbnails_io, format="PNG")
            thumbs.append((thumbnails_io.getvalue(), r["final_label"], r["final_confidence"]))
        return (f"Processed {len(img_paths)} images. Download CSV for full results.", (csv_bytes, "results.csv")), thumbs
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}", None
    finally:
        pass  # tmp remains until process exits; HF Spaces ephemeral storage okay

def handle_url(url: str, use_cnn: bool):
    if not url:
        return "No URL provided", None
    tmp = tempfile.TemporaryDirectory()
    try:
        img_paths = extract_images_from_url(url, tmp.name)
        if not img_paths:
            return "No downloadable <img> tags found at URL.", None
        results = classify_images(img_paths, use_cnn=use_cnn)
        df = pd.DataFrame(results)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        thumbs = []
        for r in results[:30]:
            pil = load_pil_image(r["image_path"])
            thumbnails_io = io.BytesIO()
            pil.resize((240,240)).save(thumbnails_io, format="PNG")
            thumbs.append((thumbnails_io.getvalue(), r["final_label"], r["final_confidence"]))
        return (f"Processed {len(img_paths)} images. Download CSV for full results.", (csv_bytes, "results.csv")), thumbs
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}", None

def handle_images_upload(images: List[io.BytesIO], use_cnn: bool):
    if not images:
        return "No images uploaded", None
    tmp = tempfile.mkdtemp()
    paths = []
    for i, img_byte in enumerate(images):
        content = img_byte.read() if hasattr(img_byte, "read") else img_byte
        p = Path(tmp) / f"upload_{i+1}.png"
        with open(p, "wb") as f:
            f.write(content)
        paths.append(str(p))
    results = classify_images(paths, use_cnn=use_cnn)
    df = pd.DataFrame(results)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    thumbs = []
    for r in results[:30]:
        pil = load_pil_image(r["image_path"])
        thumbnails_io = io.BytesIO()
        pil.resize((240,240)).save(thumbnails_io, format="PNG")
        thumbs.append((thumbnails_io.getvalue(), r["final_label"], r["final_confidence"]))
    return (f"Processed {len(paths)} uploaded images. Download CSV for full results.", (csv_bytes, "results.csv")), thumbs

# -------------------------
# Gradio Layout
# -------------------------
with gr.Blocks(title="Medical vs Non-Medical Image Classifier") as demo:
    gr.Markdown("# Medical vs Non-Medical Image Classifier")
    gr.Markdown("Upload a PDF, provide a webpage URL, or upload images. Uses CLIP zero-shot + optional MobileNetV2 CNN.")
    with gr.Row():
        with gr.Column():
            pdf_in = gr.File(label="Upload PDF (extract images from PDF)", file_types=[".pdf"])
            url_in = gr.Textbox(label="Or enter web page URL (will download <img> tags)", placeholder="https://example.com")
            img_upload = gr.Files(label="Or upload image files (jpg/png)", file_count="multiple")
            use_cnn_checkbox = gr.Checkbox(label="Use fine-tuned CNN (if best_model.pth present)", value=True)
            run_pdf = gr.Button("Classify PDF")
            run_url = gr.Button("Classify URL")
            run_img = gr.Button("Classify Uploaded Images")
            status = gr.Markdown("", visible=True)
        with gr.Column():
            csv_download = gr.File(label="Download results CSV", interactive=False)
            gallery = gr.Gallery(label="Top results (thumbnails)").style(grid=[3], height="auto")

    # Callbacks
    run_pdf.click(fn=handle_pdf_upload, inputs=[pdf_in, use_cnn_checkbox], outputs=[status, gallery])
    run_url.click(fn=handle_url, inputs=[url_in, use_cnn_checkbox], outputs=[status, gallery])
    run_img.click(fn=handle_images_upload, inputs=[img_upload, use_cnn_checkbox], outputs=[status, gallery])

demo.launch(server_name="0.0.0.0", server_port=7860)
