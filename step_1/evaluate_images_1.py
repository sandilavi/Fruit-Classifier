import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil

# Load the trained model
model = load_model('step_1/ml_model_1.h5')

# Define the classes
class_labels = ['fruit', 'non_fruit']

# Path to the dataset folder
dataset_path = '/content/drive/My Drive/Fruit Classifier Images/step_1/dataset_1'

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    true_class = os.path.basename(os.path.dirname(img_path))

    # Get the confidence for the actual class
    true_class_idx = class_labels.index(true_class)
    true_class_confidence = prediction[true_class_idx]

    return true_class, predicted_class, img_path, prediction, true_class_confidence
    
# List to store misclassified images
misclassified_images = []

# Dictionary to store misclassified images count per class
misclassified_count_per_class = {label: 0 for label in class_labels}
iterate_count = 0

# Remove misclassified_images folder if it exists
misclassified_folder = 'step_1/misclassified_images_1'
if os.path.exists(misclassified_folder):
    shutil.rmtree(misclassified_folder)

# Create a fresh folder
os.makedirs(misclassified_folder)

# Iterate through the dataset
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check image formats
                img_path = os.path.join(folder_path, img_file)
                true_class, predicted_class, img_path, prediction, true_class_confidence = predict_image(img_path)
                
                # Check if the image was misclassified
                if true_class != class_labels[predicted_class]:
                    misclassified_images.append({
                        'image': img_path,
                        'true_class': true_class,
                        'predicted_class': class_labels[predicted_class],
                        'raw_predictions': prediction,
                        'confidence_predicted_class': prediction[predicted_class],
                        'confidence_true_class': true_class_confidence
                    })
                    
                    # Update misclassified count for the true class
                    misclassified_count_per_class[true_class] += 1

                    # Generate the new filename with predicted class
                    filename, ext = os.path.splitext(os.path.basename(img_path))
                    new_filename = f"{filename} ({class_labels[predicted_class]}){ext}"
                    new_path = os.path.join(misclassified_folder, new_filename)

                    # Copy the misclassified image
                    shutil.copy(img_path, new_path)
                    print(f"Copied Misclassified Image: {img_path} -> {new_path}")

                iterate_count += 1

print(f"{iterate_count} images have been checked for misclassification.\n")

# Print the misclassified images
for item in misclassified_images:
    print(f"Misclassified Image: {item['image']}")
    print(f"Predicted Class: {item['predicted_class']}, True Class: {item['true_class']}, "
          f"Raw predictions: {item['raw_predictions']}")
    print(f"Confidence for Predicted Class: {item['confidence_predicted_class']:.2f}, "
          f"Confidence for True Class: {item['confidence_true_class']:.2f}\n")

print(f"Total Misclassified Images: {len(misclassified_images)}\n")

# Print misclassified image counts per class
print("Misclassified Images Count Per Class:")
for label, count in misclassified_count_per_class.items():
    print(f"{label}: {count}")
