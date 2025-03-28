import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load and Preprocess Data
train_dir = "dataset/train"
val_dir = "dataset/val"

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode="binary"
)

# Step 2: Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary Classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Step 3: Train Model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Step 4: Save Model
model.save("models/plant_disease_model.h5")
print("✅ Model Training Complete & Saved!")
