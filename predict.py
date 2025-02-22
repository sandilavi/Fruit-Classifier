from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('ml_model.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(model.predict(img_array))

    # Predict the class
    prediction = model.predict(img_array)
    if prediction[0] > 0.7:
        print("This is a banana!")
    else:
        print("This is not a banana!")

# Test with a new image
predict_image("Image_1.jpg")
