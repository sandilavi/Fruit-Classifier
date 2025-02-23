from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load a pre-trained model (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers (so they won’t be updated during training)
for layer in base_model.layers:
    layer.trainable = False

# Add own layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(2, activation='softmax')(x)  # Output layer for 2 classes (banana, apple)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
    rescale=1./255, validation_split=0.2, brightness_range=[0.8, 1.2], channel_shift_range=0.2
)

# Load Augmented Data for Training & Validation
train_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="training"
)
val_generator = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="categorical", subset="validation"
)

# Train the model
# model.fit(train_generator, validation_data=val_generator, epochs=7)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(train_generator, validation_data=val_generator, epochs=7, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('fruit_classifier.h5')
