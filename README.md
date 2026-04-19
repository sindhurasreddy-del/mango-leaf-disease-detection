# Mango Leaf Disease Detection using CNN

Automatic detection and classification of mango leaf diseases using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras. The model classifies mango leaf images into **8 categories** and provides **curative steps** for each detected disease, helping farmers and agricultural practitioners take timely action.

---

##  Project Overview

Mango is one of the most widely cultivated fruits, and leaf diseases significantly impact crop yield and quality. This project automates disease identification from leaf images using deep learning, eliminating the need for manual expert inspection.

The system takes a mango leaf image as input, classifies it into one of 8 disease categories, and provides actionable treatment recommendations.

---

##  Disease Classes

| # | Class | Type | Curative Action |
|---|-------|------|-----------------|
| 1 | Anthracnose | Fungal | Fungicides (chlorothalonil, copper, mancozeb), pruning affected parts |
| 2 | Bacterial Canker | Bacterial | Copper-based bactericides, remove infected plants, sanitize tools |
| 3 | Cutting Weevil | Pest | Insecticides (pyrethroids, neem oil), beneficial nematodes |
| 4 | Die Back | Fungal | Pruning, fungicides, improve air circulation and drainage |
| 5 | Gall Midge | Pest | Insecticides, sticky traps, remove galls |
| 6 | Healthy | — | Regular monitoring, balanced nutrition, mulching |
| 7 | Powdery Mildew | Fungal | Sulfur/neem oil fungicides, increase airflow, avoid wetting foliage |
| 8 | Sooty Mould | Fungal | Control sap-sucking insects, wash leaves, insecticidal soap |

---

## Dataset

Due to GitHub file size limitations, the dataset is hosted on Google Drive.

** [Download Dataset](https://drive.google.com/drive/folders/1zOr-h6tD2SlOFVfxw2-xXdSIbP4zMhVp?usp=drive_link)**

The dataset is organized in folders by class name, with each folder containing leaf images for that disease category.

---

##  Model Architecture

```
Input (224 × 224 × 3)
    ↓
Resize & Rescale (normalize to 0-1)
    ↓
Conv2D(32, 3×3) + ReLU → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) + ReLU → MaxPool(2×2)
    ↓
Conv2D(128, 3×3) + ReLU → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) + ReLU → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) + ReLU → MaxPool(2×2)
    ↓
Flatten
    ↓
Dense(128) + ReLU → Dropout(0.5)
    ↓
Dense(64) + ReLU
    ↓
Dense(8) + Softmax → Predicted Class
```

---

## 🔧 Methodology

### Data Preprocessing
- Image resizing to **224 × 224** pixels
- Pixel normalization (rescaling to 0–1)
- Data augmentation: random horizontal/vertical flip and random rotation (0.2)

### Dataset Split
- **Training**: 80%
- **Validation**: 10%
- **Test**: 10%
- Efficient data pipeline using caching, shuffling, and prefetching (`tf.data.AUTOTUNE`)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metric**: Accuracy
- **Batch Size**: 32
- **Epochs**: 3
- **Image Size**: 224 × 224 × 3

---

## 📈 Evaluation

The model is evaluated using:
- **Training & Validation Accuracy/Loss curves** — to monitor overfitting
- **Confusion Matrix** — to visualize per-class prediction performance
- **Classification Report** — Precision, Recall, and F1-score for each disease class

---

## Prediction & Cure Recommendation

The trained model takes a user-provided leaf image and outputs:
1. **Predicted disease name**
2. **Confidence score (%)**
3. **Curative steps** — specific treatment actions (fungicides, pruning, pest control, etc.)

Example output:
```
Predicted: Anthracnose, Confidence: 94.5%

Curative steps:
- Remove affected parts: Prune and destroy affected leaves and plant parts.
- Apply fungicides: Use fungicides containing chlorothalonil, copper, or mancozeb.
- Maintain sanitation: Clean the area around plants of plant debris.
- Water properly: Avoid overhead watering.
```

---

##  Project Structure

```
mango-leaf-disease-detection/
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore rules
├── notebooks/
│   └── mango_disease_detection.ipynb  # Main Jupyter notebook
├── src/
│   └── predict.py                   # Prediction script with cure recommendations
├── saved/
│   └── model.h5                     # Trained model (generated after training)
└── images/
    └── sample_predictions.png       # Sample prediction screenshots
```

---

##  How to Run

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Step 1 — Download the Dataset
Download from [Google Drive](https://drive.google.com/drive/folders/1zOr-h6tD2SlOFVfxw2-xXdSIbP4zMhVp?usp=drive_link) and extract to a local folder.

### Step 2 — Train the Model
Open the Jupyter notebook and run all cells:
```bash
jupyter notebook notebooks/mango_disease_detection.ipynb
```

### Step 3 — Make Predictions
```python
from src.predict import load_and_predict_with_cure_steps

load_and_predict_with_cure_steps(
    model_path="saved/model.h5",
    image_path="path/to/your/leaf_image.jpg"
)
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Model Format | HDF5 (.h5) |

---

##  Future Improvements

1. **Increase training epochs** — Current model trains for only 3 epochs; more epochs with early stopping would improve accuracy
2. **Transfer learning** — Use pre-trained models (ResNet50, EfficientNet) for better feature extraction
3. **Web deployment** — Build a Flask/Streamlit app for real-time leaf disease detection
4. **Mobile app** — Convert model to TensorFlow Lite for on-device inference
5. **Larger dataset** — Collect more diverse images across different lighting conditions and leaf ages

---

