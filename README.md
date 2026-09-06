# # ♻️ EcoClassify — AI-Powered Waste Classification System🗑️♻️

> **B.Tech CSE (IoT) Academic & Portfolio Project**  
> AI-powered waste identification using Deep Learning (MobileNetV2 + Transfer Learning)

---

## Overview

This project is a complete AI-powered **waste classification system** that uses a Convolutional Neural Network (CNN) with Transfer Learning to automatically identify waste type from a photograph.

A user uploads an image of waste through a professional web interface, and the system instantly predicts which of six categories the waste belongs to, along with the prediction confidence.

---

## Problem Statement

Manual waste sorting is time-consuming, error-prone, and expensive. Misclassified waste leads to contamination of recyclable materials, increased landfill usage, and environmental harm.

An automated visual waste classification system can:
- Speed up waste sorting in recycling facilities
- Reduce human error in waste management
- Enable smart waste bins for IoT applications
- Provide real-time guidance to households on proper waste disposal

---

## Objectives

1. Build a deep learning model that classifies waste images into 6 categories
2. Apply transfer learning (MobileNetV2 + ImageNet weights) for strong performance with limited data
3. Handle class imbalance with stratified splitting and class weighting
4. Deliver a professional Streamlit web application for real-world usage
5. Provide a complete, reproducible, beginner-friendly codebase

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNetV2 | Pretrained CNN backbone |
| Pillow | Image loading and processing |
| NumPy | Numerical computation |
| Pandas | Data handling |
| Matplotlib | Training curve plots |
| Seaborn | Confusion matrix visualisation |
| Scikit-learn | Train/test split, metrics |
| Streamlit | Web application UI |

---

## Dataset

The dataset contains **2,527 real-world waste images** across 6 categories:

| Class | Images |
|---|---|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

Dataset path: `Dataset/extracted/dataset-resized/`

> ⚠️ The `trash` class is significantly smaller (137 images). The training pipeline uses **class weighting** and **stratified splitting** to handle this imbalance.

---

## Project Architecture

```
Image Upload
     ↓
Preprocessing (resize 224×224, normalize for MobileNetV2)
     ↓
MobileNetV2 Base (ImageNet pretrained, frozen → fine-tuned)
     ↓
Global Average Pooling
     ↓
Dropout (0.4)
     ↓
Dense 128 (ReLU)
     ↓
Dense 6 (Softmax)
     ↓
Predicted Class + Confidence Score
     ↓
Streamlit Web Interface
```

---

## Project Structure

```
waste classification/
│
├── Dataset/
│   └── extracted/
│       └── dataset-resized/      ← 2,527 images across 6 classes
│
├── models/
│   ├── waste_classifier.keras    ← saved best model
│   └── data_split.json           ← reproducible train/val/test split
│
├── outputs/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset_analysis.py       ← inspect & verify the dataset
│   ├── train.py                  ← full training pipeline
│   ├── evaluate.py               ← model evaluation
│   └── predict.py                ← single-image CLI prediction
│
├── app.py                        ← Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/EcoClassify.git
cd EcoClassify
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Python 3.14:** `tf-nightly` is required because TensorFlow's stable release does not yet support Python 3.14. `requirements.txt` already includes it.

---

## Dataset Setup

The dataset should be placed at:

```
Dataset/extracted/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Verify the dataset:

```bash
python src/dataset_analysis.py
```

---

## Training

```bash
python src/train.py
```

Training runs in two phases:
- **Phase A** (up to 15 epochs): Classification head trained, MobileNetV2 base frozen
- **Phase B** (up to 10 epochs): Top 30 layers of MobileNetV2 unfrozen for fine-tuning

Best model is automatically saved to `models/waste_classifier.keras`.  
Training curves are saved to `outputs/`.

---

## Evaluation

```bash
python src/evaluate.py
```

Outputs:
- Test accuracy printed in terminal
- `outputs/classification_report.txt` — precision, recall, F1 per class
- `outputs/confusion_matrix.png` — visual confusion matrix

---

## Prediction (single image)

```bash
python src/predict.py path/to/image.jpg
```

Example output:
```
========================================
  Image           : bottle.jpg
  Predicted class : Plastic
  Confidence      : 94.32%
========================================
```

---

## Running the Web Application

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Expected Output

The Streamlit application allows you to:
1. Upload any waste image (JPG, JPEG, PNG)
2. See the uploaded image displayed
3. Receive the predicted waste category with confidence %
4. View a probability bar chart for all 6 categories
5. Read disposal guidance for the predicted category

---

## Future Enhancements

| Enhancement | Description |
|---|---|
| 📷 Real-time camera | Live classification via webcam feed |
| 🗑️ Smart bin IoT | Automatic bin lid control based on prediction |
| 📊 Cloud dashboard | Waste analytics and monitoring dashboard |
| 🌍 GPS collection | Location-based waste collection routing |
| ♻️ Recycling guide | Localised recycling recommendations by city |
| 📱 Mobile app | Android/iOS app for household use |
| 🔬 Edge AI | Deploy on Raspberry Pi / NVIDIA Jetson for embedded bins |
| 📦 Quantity monitoring | Estimate how full bins are using image analysis |

---

## Academic Use

This project is suitable as a **B.Tech CSE (IoT)** final year or semester project.  
Key talking points for presentation/interview:

- Transfer learning and why it's beneficial with limited data
- Class imbalance handling (stratified split + class weights)
- Two-phase training: frozen base → fine-tuning
- MobileNetV2 architecture advantages (lightweight, mobile-ready)
- End-to-end deployment with Streamlit

---

## License

This project is intended for academic and educational purposes.
