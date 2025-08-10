# infer.py
import argparse, os
from PIL import Image
import torch
from models.clip_inference import clip_zero_shot_predict
from models.cnn_model import get_model
from utils.preprocess import get_transforms

def run_inference_folder(folder, cnn_weights=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
    image_paths.sort()
    # Load CNN if given
    cnn_model = None
    transform = get_transforms(img_size=224, is_train=False)
    if cnn_weights:
        cnn_model = get_model(num_classes=2, base_model="mobilenet_v2", pretrained=False)
        cnn_model.load_state_dict(torch.load(cnn_weights, map_location=device))
        cnn_model.to(device)
        cnn_model.eval()

    results = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        print("Image:", p)
        # CLIP
        try:
            clip_res = clip_zero_shot_predict(img, device=device)
            clip_pred = clip_res['labels'][int(clip_res['probs'].argmax())]
            print("CLIP:", list(zip(clip_res['labels'], clip_res['probs'])))
        except Exception as e:
            print("CLIP failed:", e)
            clip_pred = None
        # CNN
        cnn_pred = None
        if cnn_model is not None:
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = cnn_model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                # index 1 is medical in our training convention
                cnn_pred = ("medical" if probs[1] > probs[0] else "non-medical", float(max(probs)))
                print("CNN:", cnn_pred)
        print("-"*40)
        results.append({"image": p, "clip": clip_pred, "cnn": cnn_pred})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--cnn_weights", default=None)
    args = parser.parse_args()
    run_inference_folder(args.folder, args.cnn_weights)
