from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('fruit_classifier.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]  # Extract prediction array
    print(f"Raw predictions: {prediction}")

    banana_confidence = prediction[0]  # Confidence for banana
    apple_confidence = prediction[1]  # Confidence for apple

    if max(banana_confidence, apple_confidence) < 0.5:  
        print("This is something else.")
        print(f"Confidence: {1 - max(banana_confidence, apple_confidence):.2%}")
    elif banana_confidence > apple_confidence:
        print("This is a banana!")
        print(f"Confidence: {banana_confidence:.2%}")
    else:
        print("This is an apple!")
        print(f"Confidence: {apple_confidence:.2%}")

# Test with a new image
predict_image("image_banana.jpg")
#predict_image("image_apple.jpg")
#predict_image("image_other.jpeg")
