import os
import pathlib
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta

dataset_path = r'/home/kvp5640/THESIS/dataset_fst'
data_dir = pathlib.Path(dataset_path)

batch_size = 32
img_height = 299    
img_width = 299
os.chdir(dataset_path)

data_gen = ImageDataGenerator(
    rotation_range=10,  
    zoom_range=0.2, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True,
    vertical_flip=True
)

# training data
train_ds = data_gen.flow_from_directory(
    data_dir / 'train',  
    target_size =(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / 'validation',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    seed=123
)

# test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / 'test',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False, 
)

print("Class indices:", train_ds.class_indices)

# Extract class names in the same order as the labels
class_names = list(train_ds.class_indices.keys())
print("Dataset class names in order:", class_names)

base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(img_height, img_width, 3))

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Get the output from the 'mixed4' layer
x = base_model.get_layer('mixed4').output

# Add custom layers after the 'mixed4' layer
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(5, activation='softmax')(x)  

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))

# optimizer
optimizer=Adadelta(learning_rate=1.0)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() 
# Train the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=50,
                    batch_size=batch_size)

results = model.evaluate(test_ds)
print(f"Test loss: {results[0]}")
print(f"Test accuracy: {results[1]}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on the test dataset
y_pred = model.predict(test_ds, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Get true labels from test dataset (one-hot encoded)
y_true = []
for images, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot labels to class labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print classification report which includes per-class accuracy, precision, recall, F1-score
print("Classification Report:\n", classification_report(y_true, y_pred))

# Define your class names (replace these with your actual class names)
class_names = ['ants', 'bed_bugs', 'chiggers', 'mosquitos', 'ticks']  

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Print class names with their accuracies
print("Per-Class Accuracy:")
for i, accuracy in enumerate(class_accuracy):
    print(f"{class_names[i]}: {accuracy:.2f}")
