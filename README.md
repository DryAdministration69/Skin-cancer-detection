# Melanoma Skin Cancer Classification & EDA App

This project performs Exploratory Data Analysis (EDA) and builds a Convolutional Neural Network (CNN) classifier to detect melanoma skin cancer directly from structured image directories (train/test folders containing benign and malignant classes). The web application is built using **Streamlit** for interactive deployment.

## Dataset
* **Source:** [Kaggle - Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
* **Structure:** Directory-based image dataset (no CSV needed)
  * `train/benign` & `train/malignant`
  * `test/benign` & `test/malignant`
* **Classes:** Benign vs. Malignant

## Project Structure
```
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── melanoma_cnn_model.h5      # Trained CNN model weights
├── class_distribution_pie_chart.png # Dataset class breakdown
├── dataset_distribution_bar.png     # Train/test split counts
├── confusion_matrix.png       # Model evaluation confusion matrix
├── roc_curve.png              # ROC curve and AUC score
└── README.md                  # Project documentation
```

## Model Architecture
The custom CNN is built using TensorFlow/Keras and features:
* Four Convolutional Blocks with ReLU activation and Max Pooling.
* Data Augmentation pipeline (Random Flip, Rotation, Zoom).
* Dropout regularization (0.5) to prevent overfitting.
* Dense classification head with Sigmoid activation for binary classification.

## Local Installation & Running
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-skin-cancer-detection.git
   cd melanoma-skin-cancer-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
