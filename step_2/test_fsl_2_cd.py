import pickle
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set parameters
IMAGE_SIZE = (224, 224)
TEST_IMAGE = "testing_images/apple_32.jpg"  # Path to the test image

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

# Load the test image for bounding box detection
img = cv2.imread(TEST_IMAGE)
orig_img = img.copy()

# Convert to grayscale and apply Gaussian Blur
gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding for object segmentation
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and draw green bounding boxes
for cnt in contours:
    if cv2.contourArea(cnt) > 500:  # Ignore small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

# Calculate confidence as percentage
confidence_percentage = highest_similarity * 100

# Display results with matplotlib
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"{best_match} ({confidence_percentage:.2f}%)")
plt.axis('off')  # Hide axes
plt.show()
