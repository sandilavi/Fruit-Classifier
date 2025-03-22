import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Set parameters
IMAGE_SIZE = (224, 224)
TEST_IMAGE = "testing_images/apple_32.jpg"

# Load YOLOv8 model
detector = YOLO("yolov8n.pt")

# Load VGG16 model
feature_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Load saved fruit feature embeddings
with open("step_2/features_2.pkl", "rb") as f:
    feature_dict = pickle.load(f)

def extract_features(image, model):
    """Extracts VGG16 features from an image."""
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def classify_object(roi):
    """Classify detected object using VGG16 features and cosine similarity."""
    roi_resized = cv2.resize(roi, IMAGE_SIZE)
    test_feature = extract_features(roi_resized, feature_model)
    best_match, highest_similarity = "Unknown", 0

    for class_name, features in feature_dict.items():
        for ref_feature in features:
            similarity = cosine_similarity([test_feature], [ref_feature])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = class_name

    return best_match, highest_similarity * 100  # Convert to percentage

# Read image
image = cv2.imread(TEST_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform object detection
detections = detector(image_rgb)

best_fruit_type = "Unknown"
best_confidence = 0

# Process detections
for result in detections:
    for det in result.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf[0].item()  # YOLO confidence
        cls = int(det.cls[0])  # YOLO class index
        label = detector.names[cls]  # YOLO label
        roi = image_rgb[y1:y2, x1:x2]

        if roi.size > 0:
            predicted_class, classification_conf = classify_object(roi)
            if classification_conf > best_confidence:
                best_fruit_type = predicted_class
                best_confidence = classification_conf

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{predicted_class} ({classification_conf:.2f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convert for matplotlib display
image_rgb_for_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display final result
plt.figure(figsize=(4, 4))
plt.imshow(image_rgb_for_display)
plt.axis('off')
plt.show()
