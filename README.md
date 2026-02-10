# mango-leaf-disease-detection
## Project Overview**

This project focuses on automatic detection and classification of mango leaf diseases using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras.
The model classifies mango leaf images into 8 categories and also provides recommended curative steps for each detected disease, helping farmers and agricultural practitioners take timely action.

## 📂 Dataset

Due to GitHub file size limitations, the dataset is hosted on Google Drive.

🔗 **Download Dataset:** 

https://drive.google.com/drive/folders/1zOr-h6tD2SlOFVfxw2-xXdSIbP4zMhVp?usp=drive_link



## Objectives**

Build a deep learning model to classify mango leaf diseases from images

Improve crop health monitoring through automation

Provide disease-specific cure recommendations after prediction

## Disease Classes**

The model classifies mango leaf images into the following categories:

*Anthracnose

*Bacterial Canker

*Cutting Weevil

*Die Back

*Gall Midge

*Healthy

*Powdery Mildew

*Sooty Mould

## Methodology

Dataset Loading

Images are loaded from directory structure using image_dataset_from_directory

Automatic label assignment based on folder names

Data Preprocessing

Image resizing to 224 × 224

Pixel normalization

Data augmentation (random flip & rotation)

Dataset Split

Training set: 80%

Validation set: 10%

Test set: 10%

Efficient pipeline using caching, shuffling, and prefetching

Model Architecture

Multiple Conv2D and MaxPooling layers

Fully connected dense layers

Dropout for regularization

Softmax output layer for multi-class classification

Training

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metric: Accuracy

Evaluation

Accuracy and loss curves

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

## Disease Prediction & Cure Recommendation

The trained model predicts the disease from a user-provided leaf image

Displays:

Predicted disease name

Confidence score (%)

Corresponding curative steps (fungicides, pruning, pest control, etc.)

## Model Performance

Evaluated using:

Test dataset accuracy

Confusion matrix visualization

Classification report for detailed class-wise performance

## Technologies Used

Python

TensorFlow & Keras

NumPy, Pandas

Matplotlib & Seaborn

Scikit-learn

## Output

Trained model saved as .h5

Prediction visualization with confidence score

Disease-specific treatment guidance
