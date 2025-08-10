# Medical vs Non-Medical Image Classifier

## Quick overview
- Extract images from PDFs & web pages.
- Two inference paths:
  - CLIP zero-shot (no training)
  - Lightweight CNN (MobileNetV2 for featre extraction) ad K-Means for clustering 

## Setup
1. Create venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
