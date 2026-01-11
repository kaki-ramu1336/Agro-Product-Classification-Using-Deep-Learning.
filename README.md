# Agro-Product-Classification-Using-Deep-Learning.

# Project Overview

* This project focuses on building a deep learning–based image classification system to automatically classify agricultural products into different categories using images. It helps reduce manual effort and improves accuracy in agro supply-chain operations.

# Objective

* To classify agro product images using a CNN model

* To automate product identification

* To demonstrate a real-world deep learning application

#  Problem Statement

* Manual classification of agricultural products from images is time-consuming and error-prone. An automated deep learning solution is required for accurate and efficient classification.

# Dataset

* Image dataset with class-wise folders

* Each folder represents a product category

* Sample classes include:

     * Tomato

     * Onion

     * Potato

     * Indian Market

# Folder Structure:

datasets/
│
├── train/
│   ├── indian_market/
│   ├── onion/
│   ├── potato/
│   └── tomato/
│
└── test/
    ├── indian_market/
    ├── onion/
    ├── potato/
    └── tomato/


# Tools & Technologies Used:

* Python

* TensorFlow / Keras

* NumPy

* Matplotlib

* Seaborn

* Scikit-learn

# Model Architecture

* Three Convolutional layers

* MaxPooling layers

* Fully connected Dense layer

* Dropout for regularization

* Softmax output layer for multi-class classification

# Model Performance:
*  Training Accuracy: ~81%

* Test Accuracy: ~74%

* The model performs well on dominant classes such as potato and market images, while the tomato class is limited due to very few training samples.

# Evaluation Metrics:
 * Accuracy
*  Precision
  * Recall
 * F1-score
 * Confusion Matrix
These metrics provide detailed class-wise performance analysis.

# Key Observations

* The model correctly classifies most of the test images.

* Training and validation performance are closely aligned, showing stable learning.

* Only a few misclassifications occur due to visual similarity between products.

* The model performs well on unseen data, indicating good generalization.

# Future Improvements

* Use transfer learning models (MobileNet, ResNet) to improve accuracy.

* Increase dataset size to enhance model generalization.

* Apply stronger data augmentation techniques.

* Deploy the model as a web or mobile application for real-time use.

# How to Run:
 1. Clone the repository
 
 2. Extract the dataset zip files
 
 3. Open the notebook
 
 4. Run all cells sequentially











 
