import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adadelta

dataset_path = r'/home/kvp5640/THESIS/dataset_fst'
data_dir = pathlib.Path(dataset_path)

batch_size = 64
img_height = 224
img_width = 224
os.chdir(dataset_path)

# training dataset
data_gen = ImageDataGenerator(
    rotation_range=2,           # Rotate images within [-2, 2] degree
    #zoom_range=0.2,            # Zoom by a factor of 0.2
    width_shift_range=0.05,     # Shift width by 5%
    #height_shift_range=0.05,   # Shift height by 5%
    horizontal_flip=True,       # Flip horizontally
    vertical_flip=True          # Flip vertically
)

# Apply data augmentation to training data
train_ds = data_gen.flow_from_directory(
    data_dir / 'train',    
    target_size=(img_height, img_width),
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
    shuffle=False  
)

model = Sequential()

model.add(Rescaling(1./255, input_shape=(224, 224, 3)))

# input & convolutional layer 1
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

# convolutional layer 2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

# convolutional layer 3
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2))) 

# Additional Convolutional Layer (Layer 4)
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))  
model.add(keras.layers.MaxPooling2D((2, 2))) 

# 3D to 1D output
model.add(Flatten())

# dense layer
model.add(Dense(128, activation='relu'))

# Dropout with a rate of 30% so the model drops random nuerons and prevents overfitting
model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax'))

model.summary()

METRICS = ['accuracy',
               	Precision(name='precision'),
               	Recall(name='recall')]

# Convert labels to one-hot encoding
# train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=len(class_names))))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = METRICS)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define class labels and their counts
class_labels = [0, 1, 2, 3, 4]  
class_counts = [169, 182, 173, 149, 206]

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array(class_labels), y=np.repeat(class_labels, class_counts))
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    batch_size=64,
    class_weight=class_weights
)

print(history.history.keys())

results  = model.evaluate(test_ds)
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

class_names = ['ants', 'bed_bugs', 'chiggers', 'mosquitos', 'ticks'] 

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Print class names with their accuracies
print("Per-Class Accuracy:")
for i, accuracy in enumerate(class_accuracy):
    print(f"{class_names[i]}: {accuracy:.2f}")

