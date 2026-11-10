# Melanoma Skin Cancer Classification & EDA App

This project performs Exploratory Data Analysis (EDA) and builds a Convolutional Neural Network (CNN) classifier to detect melanoma skin cancer directly from structured image directories (train/test folders containing benign and malignant classes). The web application is built using **Streamlit** for interactive deployment.

## Dataset
* **Source:** [Kaggle - Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
* **Structure:** Directory-based image dataset
  * `train/benign` & `train/malignant`
  * `test/benign` & `test/malignant`
* **Classes:** Benign vs. Malignant

## Project Structure
```
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies (uses tensorflow-cpu for cloud compatibility)
├── melanoma_cnn_model.h5      # Trained CNN model weights
├── class_distribution_pie_chart.png # Dataset class breakdown
├── dataset_distribution_bar.png     # Train/test split counts
├── confusion_matrix.png       # Model evaluation confusion matrix
├── roc_curve.png              # ROC curve and AUC score
└── README.md                  # Project documentation
```

## Local Installation & Running
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
