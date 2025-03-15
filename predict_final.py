import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# Load trained models
model_fruit_non_fruit = load_model('step_1/ml_model_1.h5')  # Step 1
model_fruit_type = load_model('step_2/ml_model_2.h5')  # Step 2

# Function to generate Grad-CAM heatmap
def generate_grad_cam(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.input], 
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))  # Get mean of gradients
    pooled_grads = tf.reshape(pooled_grads, (1, 1, 1, -1))  # Reshape to match conv_outputs

    heatmap = conv_outputs * pooled_grads  # Correct shape match
    heatmap = tf.reduce_mean(heatmap, axis=-1)[0]  # Reduce across filters

    heatmap = np.maximum(heatmap, 0)  # ReLU: keep positive values
    heatmap /= np.max(heatmap)  # Normalize
    return heatmap

# Function to overlay bounding boxes
def draw_bounding_boxes(img_path, model_fruit_non_fruit, model_fruit_type):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img.copy()

    # Preprocess image
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Step 1: Fruit vs Non-Fruit
    prediction = model_fruit_non_fruit.predict(img_array)[0]
    class_labels = ['fruit', 'non_fruit']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Store predictions
    predictions = [f"{predicted_class} ({confidence*100:.2f}%)"]
    
    # If the object is a fruit, classify the type
    if predicted_class == 'fruit':
        # Step 2: Fruit Identification
        predicted_type, type_confidence = predict_fruit_type(img_array)
        predictions.append(f"Fruit Type: {predicted_type} ({type_confidence*100:.2f}%)")

        # Generate Grad-CAM for fruit detection
        heatmap = generate_grad_cam(img_array, model_fruit_type, 'block5_conv3')  # Adjust layer for fruit type model
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap_resized)

        # Apply threshold to find most activated area
        _, heatmap_thresh = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(heatmap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw red box for the model's focus area (for both classes)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Model's focus area (red)

        # Detect actual fruit using optimized contour detection and draw green bounding box for fruit only
        gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better fruit segmentation
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours based on the new thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to ignore small regions
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Ignore small objects
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Actual fruit area (green)

    else:
        # If it's a non-fruit, draw red bounding box (but no green box)
        gray = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better non-fruit segmentation
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours based on the new thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to ignore small regions
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Ignore small objects
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Non-fruit area (red)

    # Print predictions
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{', '.join(predictions)}")
    plt.show()

# Function to predict fruit type
def predict_fruit_type(img_array):
    prediction = model_fruit_type.predict(img_array)[0]
    fruit_types = ['apple', 'banana', 'pineapple']
    predicted_type = fruit_types[np.argmax(prediction)]
    return predicted_type, np.max(prediction)

# Test with an image
image_path = os.path.join("step_1", "dataset_1", "fruit", "fruit_1.jpg")
#image_path = os.path.join("testing_images", "OIP2.jpeg")
draw_bounding_boxes(image_path, model_fruit_non_fruit, model_fruit_type)
