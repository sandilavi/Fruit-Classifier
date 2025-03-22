import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Set parameters
IMAGE_SIZE = (224, 224)
TEST_IMAGE = "testing_images/apple_37.jpeg"  # Path to the test image

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

# Load Features
with open("step_2/features_2.pkl", "rb") as f:
    feature_dict = pickle.load(f)

# Extract features from test image
test_feature = extract_features(TEST_IMAGE, model)

# Compare test image with reference images using cosine similarity
best_match = None
highest_similarity = -1

for class_name, features in feature_dict.items():
    for ref_feature in features:
        similarity = cosine_similarity([test_feature], [ref_feature])[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = class_name

print(f"Predicted Class: {best_match} (Confidence: {highest_similarity:.2f})")
