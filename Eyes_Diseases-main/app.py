import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(page_title="Eye Disease Detection", page_icon="👁️", layout="centered")

# Title and description
st.title("👁️ Eye Disease Detection")
st.write("Upload an eye image to detect Cataract, Diabetic Retinopathy, or Glaucoma")


# Load model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_eye_model():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    weights_path = "model_weights.h5"

    # Check if weights file exists
    if not os.path.exists(weights_path):
        st.error(f"❌ Model weights not found!")
        st.info("Make sure model_weights.h5 is in your repository")
        st.stop()

    try:
        # Manually rebuild the model architecture based on the error message
        model = keras.Sequential(
            [
                layers.Input(shape=(224, 224, 3)),
                # Block 1
                layers.SeparableConv2D(16, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                # Block 2
                layers.SeparableConv2D(32, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                # Block 3
                layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2),
                # Classifier
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.4),
                layers.Dense(
                    4, activation="softmax"
                ),  # 4 classes based on architecture
            ]
        )

        # Load the weights
        model.load_weights(weights_path)

        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# Class names for predictions (model only has 3 disease classes)
class_names = ["Cataract", "Diabetic Retinopathy", "Glaucoma"]

# Confidence threshold for "Normal" classification
CONFIDENCE_THRESHOLD = 60  # If max confidence is below this, classify as Normal

# Load the model
with st.spinner("Loading model..."):
    model = load_eye_model()

st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing image..."):
        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Resize and normalize
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Determine if it's a normal eye based on confidence threshold
        is_normal = confidence < CONFIDENCE_THRESHOLD

        if is_normal:
            predicted_disease = "Normal"
        else:
            predicted_disease = class_names[predicted_class]

    # Display results
    with col2:
        st.subheader("Prediction Results")

        # Show predicted class with styling
        st.markdown(f"### **{class_names[predicted_class]}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # Progress bar for confidence
        st.progress(float(confidence / 100))

        # Show all class probabilities
        st.write("---")
        st.write("**All Predictions:**")
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i] * 100
            st.write(f"{class_name}: {prob:.2f}%")

    # Warning/Info message
    st.info(
        "⚠️ **Disclaimer:** This is an AI prediction tool and should not replace professional medical diagnosis. Please consult an eye care professional for accurate diagnosis."
    )

    # Download results option
    if st.button("Generate Report"):
        report = f"""
        Eye Disease Detection Report
        ============================
        
        Predicted Condition: {class_names[predicted_class]}
        Confidence Level: {confidence:.2f}%
        
        All Predictions:
        """
        for i, class_name in enumerate(class_names):
            prob = predictions[0][i] * 100
            report += f"\n- {class_name}: {prob:.2f}%"

        report += "\n\nDisclaimer: This is an AI prediction and should not replace professional medical diagnosis."

        st.download_button(
            label="Download Report",
            data=report,
            file_name="eye_disease_report.txt",
            mime="text/plain",
        )

else:
    st.info("👆 Please upload an eye image to begin analysis")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write(
        """
    This application uses deep learning to detect three common eye diseases:
    
    - **Cataract**: Clouding of the eye's lens
    - **Diabetic Retinopathy**: Diabetes-related damage to the retina
    - **Glaucoma**: Optic nerve damage
    
    ### How to use:
    1. Upload an eye image (JPG, JPEG, or PNG)
    2. Wait for the analysis
    3. Review the prediction results
    4. Optionally download the report
    
    ### Note:
    Always consult a healthcare professional for proper diagnosis and treatment.
    """
    )

    st.write("---")
    st.write("**Model Information:**")
    st.write("- Input Size: 224x224")
    st.write(f"- Classes: {len(class_names)}")
    st.write("- Framework: TensorFlow/Keras")
