import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Set parameters
IMAGE_SIZE = (224, 224)
REFERENCE_DIR = "step_1/dataset_1"  # Folder containing images for each class

# Load VGG16 model
model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

def extract_features(image_path, model):
    """Extracts VGG16 features from an image."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Extract & Save Features
feature_dict = {}

for class_name in os.listdir(REFERENCE_DIR):
    class_path = os.path.join(REFERENCE_DIR, class_name)
    if os.path.isdir(class_path):
        feature_dict[class_name] = []
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            feature_vector = extract_features(image_path, model)
            feature_dict[class_name].append(feature_vector)

# Save extracted features
with open("step_1/features_1.pkl", "wb") as f:
    pickle.dump(feature_dict, f)

print("Feature extraction complete. Saved to 'features_1.pkl'.")
