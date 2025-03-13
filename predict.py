from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the trained model
model = load_model('fruit_vs_nonfruit.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    print(f"Raw predictions: {prediction}")
    print(f"Predicting for image: {img_path}")

    class_labels = ['fruit', 'non_fruit']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    if confidence < 0.5:
        print(f"Not sure what this is.")
        print(f"Confidence: {confidence:.2%}")
    else:
        print(f"This is an image of {predicted_class}!")
        print(f"Confidence: {confidence:.2%}")

# Test with a new image
predict_image("testing_images/download1.jpeg")
#predict_image("testing_images/image_banana.jpg")
#predict_image("testing_images/image_other.jpeg")
#predict_image("testing_images/banana.jpeg")
