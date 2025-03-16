from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import os
import tensorflow as tf
import numpy as np
import random
import time
start_time = time.time()

seed_value = 2
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Load a pre-trained model (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers (so they won’t be updated during training)
for layer in base_model.layers:
    layer.trainable = False

# Add own layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(3, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest', rescale=1./255, validation_split=0.25, brightness_range=[0.8, 1.2],
    channel_shift_range=0.2
)

# Load Augmented Data for Training & Validation
train_generator = datagen.flow_from_directory(
    "/content/drive/My Drive/Fruit Classifier Images/step_2/dataset_2", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="training", shuffle=False
)
val_generator = datagen.flow_from_directory(
    "/content/drive/My Drive/Fruit Classifier Images/step_2/dataset_2", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation", shuffle=False
)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[early_stopping])
print("Training Complete! Model is ready.")

# Get true labels and predictions from the validation data
y_true = val_generator.classes
y_pred = model.predict(val_generator, verbose=1)

# Convert predictions to class labels
y_pred = np.argmax(y_pred, axis=1)

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Detailed classification report
print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate the model accuracy
loss, accuracy = model.evaluate(val_generator)
loss, training_acc = model.evaluate(train_generator)
loss, validation_acc = model.evaluate(val_generator)
print(f"Model accuracy: {accuracy * 100:.2f}%.")
print(f"Training Accuracy: {training_acc * 100:.2f}%")
print(f"Validation Accuracy: {validation_acc * 100:.2f}%")

# Save the trained model to a file
model.save('step_2/ml_model_2.h5')  
print(train_generator.class_indices)

end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
