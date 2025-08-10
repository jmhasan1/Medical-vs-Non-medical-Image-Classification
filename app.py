import streamlit as st
import torch
from PIL import Image
import joblib
import clip
import torchvision.models as models
import numpy as np
import time
import sys
import os
import shutil
import pandas as pd
from pathlib import Path
import tempfile
from sklearn.cluster import KMeans

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent))

# Import existing scripts
try:
    from scripts.extract_from_pdf import extract_images
    from scripts.extract_from_url import extract_images_from_url
    from utils.preprocess import get_basic_transforms
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required scripts are in the correct directories")
    st.stop()

def get_basic_transforms_fallback(img_size=224):
    """Fallback transforms function if utils.preprocess is not available"""
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MedicalImageClassifier:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temp_dir = None
        self.mobilenet_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.kmeans_mobilenet = None
        self.kmeans_clip = None
        
    def setup_temp_directory(self):
        """Create temporary directory for processing"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = tempfile.mkdtemp(prefix="medical_classifier_")
        return self.temp_dir
    
    def cleanup_temp_directory(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @st.cache_resource(show_spinner=False)
    def load_models(_self):
        """Load all required models"""
        try:
            # Load MobileNet
            _self.mobilenet_model = models.mobilenet_v2(pretrained=True)
            _self.mobilenet_model.classifier = torch.nn.Identity()
            _self.mobilenet_model = _self.mobilenet_model.to(_self.device)
            _self.mobilenet_model.eval()
            
            # Load CLIP
            _self.clip_model, _self.clip_preprocess = clip.load("ViT-B/32", device=_self.device)
            
            # Load trained KMeans models if they exist
            if os.path.exists('models/kmeans_mobilenet.pkl'):
                _self.kmeans_mobilenet = joblib.load('models/kmeans_mobilenet.pkl')
                st.sidebar.success("✅ Pre-trained MobileNet model loaded")
            
            if os.path.exists('models/kmeans_clip.pkl'):
                _self.kmeans_clip = joblib.load('models/kmeans_clip.pkl')
                st.sidebar.success("✅ Pre-trained CLIP model loaded")
                
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def extract_images_from_input(self, input_type, input_data):
        """Extract images from PDF or URL using existing scripts"""
        extracted_dir = os.path.join(self.temp_dir, "extracted_images")
        os.makedirs(extracted_dir, exist_ok=True)
        
        try:
            if input_type == "pdf":
                # Save uploaded PDF to temp file
                temp_pdf = os.path.join(self.temp_dir, "input.pdf")
                with open(temp_pdf, "wb") as f:
                    f.write(input_data.read())
                
                # Extract images from PDF using your script
                images_extracted = extract_images(temp_pdf, extracted_dir)
                
                # Get list of extracted image paths
                image_paths = list(Path(extracted_dir).glob("*.*"))
                image_paths = [str(p) for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
                
            elif input_type == "url":
                # Extract images from URL using your script
                images_extracted = extract_images_from_url(input_data, extracted_dir)
                
                # Get list of extracted image paths
                image_paths = list(Path(extracted_dir).glob("*.*"))
                image_paths = [str(p) for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            return image_paths
            
        except Exception as e:
            st.error(f"Error extracting images: {e}")
            return []
    
    def preprocess_images(self, image_paths):
        """Preprocess extracted images"""
        processed_dir = os.path.join(self.temp_dir, "processed_images")
        os.makedirs(processed_dir, exist_ok=True)
        
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                # Basic validation - check image size
                if img.size[0] > 50 and img.size[1] > 50:  # Skip very small images
                    processed_path = os.path.join(processed_dir, os.path.basename(img_path))
                    img.save(processed_path)
                    valid_paths.append(processed_path)
            except Exception as e:
                st.warning(f"Failed to preprocess {img_path}: {e}")
        
        return valid_paths
    
    def extract_mobilenet_features(self, image_paths):
        """Extract features using MobileNet"""
        features = []
        valid_paths = []
        
        # Use existing preprocess function or fallback
        try:
            transform = get_basic_transforms()
        except:
            transform = get_basic_transforms_fallback()
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(self.device)
                    feat = self.mobilenet_model(tensor).cpu().numpy().flatten()
                    features.append(feat)
                    valid_paths.append(img_path)
                except Exception as e:
                    st.warning(f"Failed to extract MobileNet features from {img_path}: {e}")
        
        return np.array(features) if features else np.array([]), valid_paths
    
    def extract_clip_features(self, image_paths):
        """Extract features using CLIP"""
        features = []
        valid_paths = []
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    image_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                    feat = self.clip_model.encode_image(image_tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    features.append(feat.cpu().numpy().flatten())
                    valid_paths.append(img_path)
                except Exception as e:
                    st.warning(f"Failed to extract CLIP features from {img_path}: {e}")
        
        return np.array(features) if features else np.array([]), valid_paths
    
    def classify_with_pretrained_models(self, image_paths):
        """Classify images using pre-trained KMeans models"""
        if not (self.kmeans_mobilenet and self.kmeans_clip):
            return None
        
        # Extract features
        mobilenet_features, mn_paths = self.extract_mobilenet_features(image_paths)
        clip_features, clip_paths = self.extract_clip_features(image_paths)
        
        if len(mobilenet_features) == 0 or len(clip_features) == 0:
            st.error("No valid features extracted from images")
            return None
        
        # Predict with KMeans
        mn_predictions = self.kmeans_mobilenet.predict(mobilenet_features)
        clip_predictions = self.kmeans_clip.predict(clip_features)
        
        # Map predictions to labels (assuming 0=medical, 1=non-medical based on your clustering)
        mn_labels = ["medical" if p == 0 else "non-medical" for p in mn_predictions]
        clip_labels = ["medical" if p == 0 else "non-medical" for p in clip_predictions]
        
        # Create results DataFrame
        results = []
        for i in range(min(len(mn_paths), len(clip_paths))):
            # Ensemble prediction (prefer CLIP in case of disagreement)
            ensemble_pred = clip_labels[i] if mn_labels[i] != clip_labels[i] else mn_labels[i]
            confidence = 'high' if mn_labels[i] == clip_labels[i] else 'medium'
            
            results.append({
                'image_path': mn_paths[i],
                'mobilenet_prediction': mn_labels[i],
                'clip_prediction': clip_labels[i],
                'ensemble_prediction': ensemble_pred,
                'confidence': confidence
            })
        
        return pd.DataFrame(results)
    
    def train_and_classify(self, image_paths):
        """Train new models and classify images"""
        try:
            # Extract features
            mobilenet_features, mn_paths = self.extract_mobilenet_features(image_paths)
            clip_features, clip_paths = self.extract_clip_features(image_paths)
            
            if len(mobilenet_features) == 0 or len(clip_features) == 0:
                st.error("No valid features extracted for training")
                return None
            
            # Train KMeans on MobileNet features
            st.info("Training MobileNet clustering model...")
            kmeans_mn = KMeans(n_clusters=2, random_state=42, n_init=10)
            mn_predictions = kmeans_mn.fit_predict(mobilenet_features)
            
            # Zero-shot classification using CLIP
            st.info("Running CLIP zero-shot classification...")
            prompts = [
                "A diagnostic medical image such as an X-ray, MRI scan, CT scan, or ultrasound from a hospital",
                "A non-medical photograph such as landscapes, architecture, nature, animals, or everyday objects"
            ]
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                clip_predictions = []
                for feat in clip_features:
                    feat_tensor = torch.tensor(feat).unsqueeze(0).to(self.device)
                    feat_tensor /= feat_tensor.norm(dim=-1, keepdim=True)
                    feat_tensor = feat_tensor.to(text_features.dtype)
                    similarities = (feat_tensor @ text_features.T).squeeze(0).cpu().numpy()
                    pred = "medical" if np.argmax(similarities) == 0 else "non-medical"
                    clip_predictions.append(pred)
            
            # Map MobileNet cluster predictions to labels (simple heuristic)
            mn_labels = ["medical" if p == 0 else "non-medical" for p in mn_predictions]
            
            # Combine results
            results = []
            for i in range(min(len(mn_paths), len(clip_paths))):
                clip_pred = clip_predictions[i] if i < len(clip_predictions) else "unknown"
                mn_pred = mn_labels[i] if i < len(mn_labels) else "unknown"
                
                ensemble_pred = clip_pred if mn_pred != clip_pred else clip_pred
                confidence = 'high' if mn_pred == clip_pred else 'medium'
                
                results.append({
                    'image_path': mn_paths[i],
                    'mobilenet_prediction': mn_pred,
                    'clip_prediction': clip_pred,
                    'ensemble_prediction': ensemble_pred,
                    'confidence': confidence
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            st.error(f"Error in training and classification: {e}")
            return None


def main():
    st.set_page_config(
        page_title="Medical Image Classification Pipeline",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical vs Non-Medical Image Classification Pipeline")
    st.markdown("*Complete ML pipeline for automatic medical image classification from PDFs and URLs*")
    
    # Initialize classifier
    classifier = MedicalImageClassifier()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"**Device:** {classifier.device}")
    
    # Model loading section
    with st.sidebar.expander("🤖 Model Status", expanded=True):
        if classifier.load_models():
            st.success("✅ Models loaded successfully!")
            
            # Check for pre-trained clustering models
            has_pretrained = os.path.exists('models/kmeans_mobilenet.pkl') and os.path.exists('models/kmeans_clip.pkl')
            if has_pretrained:
                st.info("📋 Pre-trained clustering models found")
            else:
                st.warning("⚠️ No pre-trained models found. Will train on input data.")
        else:
            st.error("❌ Failed to load models")
            return
    
    # Main input section
    st.header("📤 Input Selection")
    
    input_type = st.radio(
        "Choose input type:",
        ["PDF Upload", "URL Input"],
        horizontal=True
    )
    
    if input_type == "PDF Upload":
        uploaded_files = st.file_uploader(
            "Upload PDF file(s) containing medical and non-medical images",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process PDF(s)", type="primary"):
                process_inputs(classifier, "pdf", uploaded_files)
                
    else:  # URL Input
        urls = st.text_area(
            "Enter URL(s) (one per line):",
            placeholder="https://example.com/medical-images\nhttps://example.com/page-with-images",
            help="Enter one or more URLs containing images to classify"
        )
        
        if urls.strip() and st.button("🚀 Process URL(s)", type="primary"):
            url_list = [url.strip() for url in urls.split('\n') if url.strip()]
            process_inputs(classifier, "url", url_list)


def process_inputs(classifier, input_type, input_data):
    """Process the inputs through the ML pipeline"""
    
    # Setup temporary directory
    classifier.setup_temp_directory()
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract images
        status_text.text("📥 Extracting images...")
        progress_bar.progress(10)
        
        all_image_paths = []
        
        if input_type == "pdf":
            for pdf_file in input_data:
                image_paths = classifier.extract_images_from_input("pdf", pdf_file)
                all_image_paths.extend(image_paths)
        else:  # URL
            for url in input_data:
                image_paths = classifier.extract_images_from_input("url", url)
                all_image_paths.extend(image_paths)
        
        if not all_image_paths:
            st.error("❌ No images found in the provided input(s)")
            return
        
        st.success(f"✅ Extracted {len(all_image_paths)} images")
        progress_bar.progress(30)
        
        # Step 2: Preprocess images
        status_text.text("🔧 Preprocessing images...")
        processed_paths = classifier.preprocess_images(all_image_paths)
        
        if not processed_paths:
            st.error("❌ No valid images after preprocessing")
            return
            
        progress_bar.progress(50)
        
        # Step 3: Classification
        status_text.text("🧠 Running classification...")
        
        # Check if we have pre-trained models
        has_pretrained = classifier.kmeans_mobilenet and classifier.kmeans_clip
        
        if has_pretrained:
            st.info("Using pre-trained clustering models...")
            results_df = classifier.classify_with_pretrained_models(processed_paths)
        else:
            st.info("Training new models on input data...")
            results_df = classifier.train_and_classify(processed_paths)
        
        progress_bar.progress(90)
        
        if results_df is not None and not results_df.empty:
            status_text.text("✅ Classification complete!")
            progress_bar.progress(100)
            
            # Display results
            display_results(results_df)
        else:
            st.error("❌ Classification failed or no results generated")
            
    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        
    finally:
        # Cleanup
        classifier.cleanup_temp_directory()


def display_results(results_df):
    """Display classification results"""
    
    st.header("📊 Classification Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", len(results_df))
    
    with col2:
        medical_count = len(results_df[results_df['ensemble_prediction'] == 'medical'])
        st.metric("Medical Images", medical_count)
    
    with col3:
        non_medical_count = len(results_df[results_df['ensemble_prediction'] == 'non-medical'])
        st.metric("Non-Medical Images", non_medical_count)
    
    with col4:
        high_conf_count = len(results_df[results_df['confidence'] == 'high'])
        st.metric("High Confidence", f"{high_conf_count}/{len(results_df)}")
    
    # Results table
    st.subheader("🔍 Detailed Results")
    
    # Filter options
    filter_pred = st.selectbox(
        "Filter by prediction:",
        ["All", "Medical", "Non-Medical"]
    )
    
    filtered_df = results_df.copy()
    if filter_pred == "Medical":
        filtered_df = filtered_df[filtered_df['ensemble_prediction'] == 'medical']
    elif filter_pred == "Non-Medical":
        filtered_df = filtered_df[filtered_df['ensemble_prediction'] == 'non-medical']
    
    # Display table
    st.dataframe(
        filtered_df[['image_path', 'ensemble_prediction', 'mobilenet_prediction', 'clip_prediction', 'confidence']],
        use_container_width=True
    )
    
    # Image gallery
    st.subheader("🖼️ Image Gallery")
    
    # Show sample images
    sample_size = min(12, len(filtered_df))
    sample_df = filtered_df.sample(n=sample_size) if len(filtered_df) > sample_size else filtered_df
    
    cols = st.columns(4)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        with cols[i % 4]:
            try:
                img = Image.open(row['image_path'])
                st.image(img, caption=f"{row['ensemble_prediction'].title()}", use_column_width=True)
                st.caption(f"Confidence: {row['confidence']}")
            except Exception as e:
                st.error(f"Cannot display image: {e}")
    
    # Download results
    st.subheader("💾 Download Results")
    
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="medical_classification_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()