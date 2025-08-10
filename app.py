import streamlit as st
import torch
from PIL import Image
import joblib
import clip
import torchvision.models as models
import numpy as np
import time
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent))

from utils.preprocess import get_basic_transforms


@st.cache_resource(show_spinner=False)
def load_mobilenet_model(device):
    """Load MobileNetV2 model for feature extraction"""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()  # Remove classification layer
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_clip_model(device, model_name="ViT-B/32"):
    """Load CLIP model"""
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def extract_mobilenet_embedding(model, image, device, img_size=224):
    """Extract embedding from image using MobileNet"""
    transform = get_basic_transforms(img_size)
    
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(device)
        embedding = model(tensor).cpu().numpy().flatten()
    
    return embedding


def extract_clip_embedding(model, preprocess, image, device):
    """Extract embedding from image using CLIP"""
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().flatten()
    
    return embedding


@st.cache_resource(show_spinner=False)
def load_models(device):
    """Load all models and kmeans clusterers"""
    try:
        # Load models
        mobilenet_model = load_mobilenet_model(device)
        clip_model, preprocess = load_clip_model(device)
        
        # Load KMeans models
        kmeans_mobilenet = joblib.load('models/kmeans_mobilenet.pkl')
        kmeans_clip = joblib.load('models/kmeans_clip.pkl')
        
        return mobilenet_model, clip_model, preprocess, kmeans_mobilenet, kmeans_clip
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please ensure you have run the clustering scripts first to generate the model files.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


def map_cluster_to_label(cluster_id, model_type="mobilenet"):
    """Map cluster IDs to human-readable labels"""
    # This mapping might need to be adjusted based on your actual clustering results
    # You may need to inspect your clustering results to determine the correct mapping
    
    if model_type == "mobilenet":
        # Assuming cluster 0 is medical and cluster 1 is non-medical
        # You might need to adjust this based on your actual results
        mapping = {0: "medical", 1: "non-medical"}
    else:  # CLIP
        # CLIP clustering already has zero-shot labeling in the original script
        # But we'll use a similar mapping for consistency
        mapping = {0: "medical", 1: "non-medical"}
    
    return mapping.get(cluster_id, f"cluster_{cluster_id}")


def ensemble_vote(pred1, pred2):
    """Ensemble voting strategy"""
    if pred1 == pred2:
        return pred1
    else:
        # Tie breaker: prefer CLIP prediction (as it has zero-shot labeling)
        return pred2


def main():
    st.title("🏥 Medical vs Non-medical Image Classification")
    st.markdown("*Unsupervised clustering approach using MobileNet + CLIP ensemble*")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"Using device: **{device}**")
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            mobilenet_model, clip_model, preprocess, kmeans_mobilenet, kmeans_clip = load_models(device)
            st.sidebar.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a medical or non-medical image for classification"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            img = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Classification Results")
                
                with st.spinner("Running MobileNet classification..."):
                    # MobileNet prediction
                    start = time.perf_counter()
                    emb_mn = extract_mobilenet_embedding(mobilenet_model, img, device)
                    pred_mn_cluster = kmeans_mobilenet.predict([emb_mn])[0]
                    pred_mn = map_cluster_to_label(pred_mn_cluster, "mobilenet")
                    t_mn = time.perf_counter() - start
                
                with st.spinner("Running CLIP classification..."):
                    # CLIP prediction
                    start = time.perf_counter()
                    emb_clip = extract_clip_embedding(clip_model, preprocess, img, device)
                    pred_clip_cluster = kmeans_clip.predict([emb_clip])[0]
                    pred_clip = map_cluster_to_label(pred_clip_cluster, "clip")
                    t_clip = time.perf_counter() - start
                
                # Ensemble prediction
                pred_ensemble = ensemble_vote(pred_mn, pred_clip)
                
                # Display results
                st.metric("MobileNet Prediction", pred_mn, f"{t_mn:.4f}s")
                st.metric("CLIP Prediction", pred_clip, f"{t_clip:.4f}s")
                st.metric("🎯 Ensemble Prediction", pred_ensemble)
                
                # Confidence indicator
                if pred_mn == pred_clip:
                    st.success("✅ Both models agree - High confidence")
                else:
                    st.warning("⚠️ Models disagree - Lower confidence (using CLIP)")
                
                # Additional info
                with st.expander("Technical Details"):
                    st.write(f"**MobileNet cluster:** {pred_mn_cluster}")
                    st.write(f"**CLIP cluster:** {pred_clip_cluster}")
                    st.write(f"**Device:** {device}")
                    st.write(f"**Image size:** {img.size}")

        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1. **MobileNet**: Extracts visual features using a pre-trained CNN
    2. **CLIP**: Uses vision-language understanding for more semantic features  
    3. **Ensemble**: Combines both predictions (CLIP takes priority in case of disagreement)
    
    ### Note:
    This is an **unsupervised** approach using clustering. Make sure you have:
    - Run `python scripts/cluster_images.py --src <data_dir> --out results.csv` 
    - Run `python scripts/clip_cluster.py --src <data_dir> --out results_clip.csv`
    - Both should generate model files in the `models/` directory
    """)


if __name__ == "__main__":
    main()