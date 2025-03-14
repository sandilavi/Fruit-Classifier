import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('fruit_vs_nonfruit.h5')

def draw_border(img_path, predicted_class):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors in Matplotlib
    
    if predicted_class == "fruit":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green border
    
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = ['fruit', 'non_fruit'][np.argmax(prediction)]
    
    print(f"You're predicting for : {img_path}")
    print(f"Raw predictions: {prediction}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {np.max(prediction):.2%}")

    draw_border(img_path, predicted_class)

# Test with an image
predict_image("fruit vs non_fruit/fruit/fruit_44.jpeg")
#predict_image("fruits/pineapple/pineapple_12.jpg")
#predict_image("testing_images/download (1).jpeg")
