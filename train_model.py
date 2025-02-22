import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Preprocess Images
dataset_path = "dataset"
target_size = (224, 224)

# Initialize counter to track the number of processed images
processed_count = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):  # Check if it's a folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, target_size)  
                processed_count += 1

print(f"Preprocessing Done! {processed_count} images resized.")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
    rescale=1./255, validation_split=0.2
)

# Load Augmented Data for Training & Validation
train_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="binary", subset="training"
)

val_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="binary", subset="validation"
)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),  # Helps reduce overfitting
    Dense(1, activation="sigmoid")
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the Model
model.fit(train_generator, validation_data=val_generator, epochs=8)
print("Training Complete! Model is ready.")

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_generator)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Evaluate training accuracy
loss, training_acc = model.evaluate(train_generator)
print(f"Training Accuracy: {training_acc * 100:.2f}%")

# Evaluate validation accuracy
loss, validation_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {validation_acc * 100:.2f}%")

# Save the trained model to a file
model.save('ml_model.h5')  
print("Model saved as 'ml_model.h5'.")
