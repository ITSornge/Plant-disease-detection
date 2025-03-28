import tensorflow as tf
import numpy as np
import cv2
import sys
import os
from firebase_store import store_prediction  # Import Firebase function

# Load trained model
model = tf.keras.models.load_model("models/plant_disease_model.h5")

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
    return img

# Check if an image path was given
if len(sys.argv) < 2:
    print("❌ Please provide an image path!")
    sys.exit(1)

image_path = sys.argv[1]  # Get image path from command line
if not os.path.exists(image_path):
    print(f"❌ Error: File '{image_path}' not found!")
    sys.exit(1)

image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(image)
class_names = ["Early Blight", "Late Blight", "Healthy"]
result = class_names[np.argmax(prediction)]

# Get image filename
image_name = os.path.basename(image_path)

# Store prediction in Firebase
store_prediction(image_name, result)

print(f"✅ Prediction: {result} (Stored in Firebase)")
