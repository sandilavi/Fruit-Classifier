import os
import cv2
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import time
start_time = time.time()

seed_value = 2
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

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
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest', rescale=1./255, validation_split=0.25, brightness_range=[0.8, 1.2],
    channel_shift_range=0.2
)

# Load Augmented Data for Training & Validation
train_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="training", shuffle=False
)
val_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation", shuffle=False
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
    Dense(3, activation="softmax")
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[early_stopping])
print("Training Complete! Model is ready.")

# Get true labels and predictions from the validation data
y_true = val_generator.classes
y_pred = model.predict(val_generator, verbose=1)

# Convert predictions to class labels
y_pred = np.argmax(y_pred, axis=1)

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Detailed classification report
print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate the model accuracy
loss, accuracy = model.evaluate(val_generator)
loss, training_acc = model.evaluate(train_generator)
loss, validation_acc = model.evaluate(val_generator)
print(f"Model accuracy: {accuracy * 100:.2f}%.")
print(f"Training Accuracy: {training_acc * 100:.2f}%")
print(f"Validation Accuracy: {validation_acc * 100:.2f}%")

# Save the trained model to a file
model.save('ml_model.h5')  
print(train_generator.class_indices)

end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
