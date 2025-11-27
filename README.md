# BRAIN TUMOR DIAGNOSIS
#### CNN-based Deep Learning + Streamlit | Detection • Classification • Segmentation

This project is an end-to-end AI system that analyzes brain MRI images and performs:
- Tumor Detection – Tumor vs No Tumor
- Tumor Classification – Predict tumor type
- Tumor Segmentation – Highlight exact tumor region
- 
Designed to assist medical professionals with faster, reliable, and automated diagnosis.

## Features
✔ Three AI Models
- Detection Model: Custom CNN (TensorFlow) – Checks whether the MRI contains a tumor
- Classification Model: Custom CNN (PyTorch) – Predicts the tumor type
- Segmentation Model: UNet (PyTorch) – Outlines the exact tumor region pixel by pixel

✔ End-to-End Pipeline
- All models integrated into a single backend pipeline
- Returns:
  - Tumor probability
  - Tumor type
  - Segmentation mask
  - Overlay visualization
  - Tumor pixel statistics

✔ Fully Interactive Streamlit Frontend
- An easy-to-use interface where users can upload an MRI, view results instantly, and download a PDF medical-style report.

Additional Highlights
- Custom loss functions for improved segmentation accuracy
- Real-time predictions using preloaded models
- Modular and clean backend code suitable for production

## How to Run
1. Install Requirements
`pip install -r requirements.txt`

2. Run Streamlit App
`streamlit run frontend/app.py`

## How to Use the System
1️⃣ Upload an MRI Image
- Go to the Diagnosis page in the Streamlit app.  
- Upload any MRI scan (PNG/JPG).

The system will automatically:
- Preprocess the image  
- Run detection  
- Run classification (only if a tumor is found)  
- Run segmentation  
- Generate an overlay mask  

2️⃣ View the Results

You will see:
- Tumor probability  
- Tumor type (if tumor exists)  
- Segmentation mask overlay  
- Tumor pixel count & statistics  
- Interactive slider to compare the original image and overlay  

3️⃣ Generate a PDF Report

On the Report page:
- Add doctor notes (optional)  
- Preview the MRI + mask + predictions  
- Click Download PDF  

The system will generate a medical-style report containing:
- MRI image  
- Tumor mask  
- Detection & classification results  
- Class probabilities  
- Tumor pixel details  
- Doctor notes
