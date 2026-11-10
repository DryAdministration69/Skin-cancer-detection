import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Melanoma Skin Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# Caching the model loading for performance
@st.cache_resource
def load_model():
    model_path = "melanoma_cnn_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Section", ["Prediction App", "Dataset EDA & Visualizations", "Model Performance"])

if app_mode == "Prediction App":
    st.title("🩺 Melanoma Skin Cancer Classification")
    st.write("Upload a skin lesion image to classify it as **Benign** or **Malignant** using our trained Convolutional Neural Network.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.subheader("Analysis Results")
            if model is None:
                st.error("Model file (`melanoma_cnn_model.h5`) not found in the repository. Please upload it to enable predictions.")
            else:
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img_height, img_width = 180, 180
                    img = image.resize((img_width, img_height))
                    img_array = np.array(img) / 255.0
                    
                    # Handle RGBA / grayscale images
                    if img_array.ndim == 2:
                        img_array = np.stack((img_array,)*3, axis=-1)
                    elif img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                        
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    prediction = model.predict(img_array)[0][0]
                    
                    # Binary classification
                    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
                    label = "Malignant" if prediction > 0.5 else "Benign"
                    
                    if label == "Malignant":
                        st.error(f"**Prediction:** {label}")
                    else:
                        st.success(f"**Prediction:** {label}")
                        
                    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                    st.info("Note: This tool is for educational and experimental purposes only and should not replace professional medical diagnosis.")

elif app_mode == "Dataset EDA & Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.write("Visualizations representing the distribution and structure of the 10,000-image Melanoma directory dataset.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        if os.path.exists("class_distribution_pie_chart.png"):
            st.image("class_distribution_pie_chart.png", use_column_width=True)
        else:
            st.info("`class_distribution_pie_chart.png` not found.")

    with col2:
        st.subheader("Dataset Split Distribution")
        if os.path.exists("dataset_distribution_bar.png"):
            st.image("dataset_distribution_bar.png", use_column_width=True)
        else:
            st.info("`dataset_distribution_bar.png` not found.")
            
    st.markdown("---")
    st.subheader("Dataset Image Directory Structure")
    st.code("""
    melanoma_cancer_dataset/
    ├── train/
    │   ├── benign/     # Benign skin lesion images
    │   └── malignant/  # Malignant skin lesion images
    └── test/
        ├── benign/     # Benign test images
        └── malignant/  # Malignant test images
    """, language="text")

elif app_mode == "Model Performance":
    st.title("📈 Model Evaluation & Metrics")
    st.write("Performance metrics and validation results from the CNN training pipeline.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_column_width=True)
        else:
            st.info("`confusion_matrix.png` not found.")

    with col2:
        st.subheader("ROC Curve")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", use_column_width=True)
        else:
            st.info("`roc_curve.png` not found.")
            
    if os.path.exists("training_performance.png"):
        st.subheader("Training Performance History")
        st.image("training_performance.png", use_column_width=True)
