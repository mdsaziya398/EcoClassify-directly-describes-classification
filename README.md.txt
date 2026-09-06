# ♻️ EcoClassify — AI-Powered Waste Classification System 🗑️

## 📌 Project Overview

**EcoClassify** is an AI-powered waste classification system that uses **Deep Learning and Computer Vision** to automatically identify different types of waste from images.

The system uses the **MobileNetV2 Convolutional Neural Network (CNN)** architecture to classify waste into six categories:

- 📦 Cardboard
- 🪟 Glass
- 🔩 Metal
- 📄 Paper
- 🛍️ Plastic
- 🗑️ Trash

The project provides a **Streamlit web application** where users can upload a waste image and receive the predicted waste category along with prediction confidence and disposal guidance.

---

## 🎯 Problem Statement

Improper waste segregation is a major environmental challenge. Manual waste classification is time-consuming, inefficient, and prone to human error.

EcoClassify addresses this problem by providing an automated image-based waste classification system that can assist users in identifying waste categories and promoting proper waste segregation.

---

## 🎯 Objectives

1. To develop an AI-based waste classification system.
2. To classify waste images into six different categories.
3. To use MobileNetV2 for efficient image classification.
4. To preprocess and analyze the waste image dataset.
5. To evaluate the performance of the trained model.
6. To provide an easy-to-use web interface using Streamlit.
7. To display prediction confidence and probability information.
8. To provide basic disposal guidance for the detected waste category.
9. To support smarter and more sustainable waste management.

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep Learning framework |
| MobileNetV2 | Image classification model |
| OpenCV / PIL | Image processing |
| NumPy | Numerical computation |
| Pandas | Dataset analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| Streamlit | Web application |
| JSON | Dataset split information |

---

## 📊 Dataset

The project uses a waste image dataset containing **2,527 images** distributed across six categories.

| Waste Category | Number of Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

### Dataset Location

```text
Dataset/
└── extracted/
    └── dataset-resized/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

---

# 🏗️ Model Architecture

EcoClassify uses **MobileNetV2**, a lightweight and efficient convolutional neural network pretrained on ImageNet.

```text
Input Image
    ↓
Resize to 224 × 224
    ↓
MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout (0.4)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Output Layer (6 neurons, Softmax)
    ↓
Predicted Waste Category
```

### Model Details

- **Base Model:** MobileNetV2
- **Pretrained Weights:** ImageNet
- **Input Size:** 224 × 224 pixels
- **Dropout:** 0.4
- **Dense Layer:** 128 neurons
- **Activation:** ReLU
- **Output Classes:** 6
- **Output Activation:** Softmax

---

# 📁 Project Structure

```text
EcoClassify-directly-describes-classification/
│
├── Dataset/
│   └── extracted/
│       └── dataset-resized/
│           ├── cardboard/
│           ├── glass/
│           ├── metal/
│           ├── paper/
│           ├── plastic/
│           └── trash/
│
├── models/
│   └── data_split.json
│
├── outputs/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset_analysis.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> **Important:** The trained `waste_classifier.keras` model is provided separately through **GitHub Release v1.0.0** because the model file is larger than GitHub's normal repository upload limit.

---

# 📥 Download the Trained Model

The trained model is available in the **EcoClassify v1.0.0 GitHub Release**.

### Steps

1. Open the **Releases** section of this repository.
2. Open **EcoClassify v1.0.0**.
3. Download `waste_classifier.keras`.
4. Open the project's `models/` folder.
5. Place `waste_classifier.keras` inside it.

The final `models` folder should look like:

```text
models/
├── waste_classifier.keras
└── data_split.json
```

The application requires the trained model to perform predictions.

---

# 🚀 Installation

## 1. Clone the Repository

Open **PowerShell** or **Command Prompt** and run:

```bash
git clone https://github.com/mdsaziya398/EcoClassify-directly-describes-classification.git
```

Then:

```bash
cd EcoClassify-directly-describes-classification
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell shows an execution policy error:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# 📂 Dataset Setup

Place the dataset inside:

```text
Dataset/extracted/dataset-resized/
```

Required folders:

```text
dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

---

# 🧪 Training the Model

To train the model:

```bash
python src/train.py
```

The trained model will be saved as:

```text
models/waste_classifier.keras
```

Training outputs can be stored in:

```text
outputs/
```

> If you use the already-trained model from Release v1.0.0, you do not need to train the model again.

---

# 📈 Model Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluation process can generate:

- Accuracy curve
- Loss curve
- Confusion matrix
- Classification report

These results are stored in:

```text
outputs/
```

---

# 🔍 Making Predictions

Run:

```bash
python src/predict.py
```

The model predicts one of:

```text
Cardboard
Glass
Metal
Paper
Plastic
Trash
```

---

# 🌐 Running the Streamlit Application

Start the application:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open that URL in your web browser.

---

# 🖥️ Application Workflow

```text
User
  ↓
Upload Waste Image
  ↓
Image Preprocessing
  ↓
MobileNetV2 Model
  ↓
Waste Classification
  ↓
Prediction + Confidence
  ↓
Disposal Guidance
```

---

# 🖼️ Application Features

### 📤 Image Upload

Users can upload waste images in:

```text
JPG
JPEG
PNG
```

### 🤖 AI-Based Classification

The MobileNetV2 model analyzes the uploaded image and predicts its waste category.

### 📊 Prediction Confidence

The application displays the confidence/probability associated with the prediction.

### ♻️ Disposal Guidance

The application provides basic guidance related to appropriate disposal or segregation.

### 💻 Simple Web Interface

The Streamlit interface allows users to interact with the AI model through a simple web application.

---

# 📊 Expected Output

For an uploaded image, the application provides output similar to:

```text
Predicted Class: Plastic

Confidence: XX.XX%
```

It also displays probability information for the supported waste categories.

---

# 🔬 Project Modules

## 1. Dataset Analysis

Analyzes the dataset and category distribution.

```text
src/dataset_analysis.py
```

## 2. Model Training

Trains the MobileNetV2-based classification model.

```text
src/train.py
```

## 3. Model Evaluation

Evaluates the trained model using classification metrics and visualizations.

```text
src/evaluate.py
```

## 4. Prediction

Performs predictions on waste images.

```text
src/predict.py
```

## 5. Streamlit Application

Provides the graphical user interface.

```text
app.py
```

---

# 🛡️ .gitignore

The project uses `.gitignore` to prevent unnecessary or sensitive files from being uploaded to GitHub.

Examples:

```text
.venv/
venv/
__pycache__/
*.pyc
.env
.vscode/
.idea/
.ipynb_checkpoints/
```

**Never commit API keys, passwords, credentials, or other sensitive information.**

---

# 📌 Important Model Note

The trained model:

```text
waste_classifier.keras
```

is distributed through **GitHub Release v1.0.0** instead of being stored directly in the repository.

This keeps the Git repository lightweight while making the trained model available for users who want to run the application.

---

# 🔮 Future Enhancements

- 📱 Mobile application support
- 🌐 Cloud deployment
- 📷 Real-time camera-based classification
- ♻️ Additional waste categories
- 📊 Advanced waste analytics
- 🌍 Regional recycling guidelines
- 🧠 Improved model accuracy
- ⚡ Faster inference
- 🗺️ Nearby recycling-center recommendations
- 📈 Environmental impact tracking

---

# 🎓 Academic Use

EcoClassify demonstrates concepts including:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Image Classification
- Transfer Learning
- TensorFlow/Keras
- MobileNetV2
- Streamlit
- Sustainable Waste Management

---

# 📜 License

This project is intended for educational and academic purposes.

---

# 👩‍💻 Author

**Mohammad Saziya**

**Project:** EcoClassify  
**Title:** AI-Powered Waste Classification System

---

# ⭐ Support the Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## ♻️ EcoClassify

**Turning Waste Images into Smart Classification for a Cleaner and More Sustainable Future.**
