from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from firebase_store import store_prediction  # Import Firebase storage function

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("models/plant_disease_model.h5")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Predict
    image = preprocess_image(file_path)
    prediction = model.predict(image)[0][0]
    result = "Healthy" if prediction < 0.5 else "Diseased"

    # Store prediction in Firebase
    store_prediction(file.filename, result)

    return f"Prediction: {result} (Stored in Firebase)"

if __name__ == "__main__":
    app.run(debug=True)
