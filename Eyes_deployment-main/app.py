from flask import Flask, render_template, request
from streamlit.models import load_model
from streamlit.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "eye_disease_cnn_model.h5"
model = load_model(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classes
CLASS_NAMES = ["Uveitis", "Normal", "Eyelid", "Conjuctivitis", "Cataract"]

# Medical Advice Dictionary
ADVICE = {
    "Normal": "Dear User, your eyes appear healthy and normal. Maintain good eye hygiene and regular checkups.",
    "Uveitis": "Uveitis detected. Please consult an eye specialist immediately to prevent vision complications.",
    "Eyelid": "Eyelid disorder detected. Consider cleaning the eyelid area, avoid rubbing, and seek medical evaluation.",
    "Conjuctivitis": "Conjunctivitis detected. Avoid touching your eyes, maintain hygiene, and consider seeing a doctor.",
    "Cataract": "Cataract indication found. Please consult an ophthalmologist for proper diagnosis and treatment options.",
}


def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    confidence = float(np.max(pred))
    class_index = np.argmax(pred)

    prediction = CLASS_NAMES[class_index]
    advice = ADVICE[prediction]

    return prediction, confidence, advice


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result, confidence, advice = predict_image(file_path)

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence * 100, 2),
        advice=advice,
        uploaded_image="uploads/" + file.filename,
    )


if __name__ == "__main__":
    app.run(debug=True)
