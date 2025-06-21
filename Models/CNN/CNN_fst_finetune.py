import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
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

#--------------------------------------------------
# DDI dataset
ddi_data_dir = pathlib.Path(r'/home/kvp5640/THESIS/ddi_dataset')

# train dataset DDI 
ddi_train_ds = tf.keras.utils.image_dataset_from_directory(
    ddi_data_dir / 'train',  
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# validation dataset  DDI 
ddi_val_ds = tf.keras.utils.image_dataset_from_directory(
    ddi_data_dir / 'validation',  
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

ddi_val_ds = ddi_val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2)))
ddi_train_ds = ddi_train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2)))

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

# Convolutional Layer 4
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))  
model.add(keras.layers.MaxPooling2D((2, 2))) 

# 3D to 1D output
model.add(Flatten())

# dense layer
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

model.summary()

METRICS = ['accuracy',
               	Precision(name='precision'),
               	Recall(name='recall')]

# Convert labels to one-hot encoding
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_ddi = tf.keras.optimizers.Adam(learning_rate=0.0001)

for layer in model.layers[:-3]:  
    layer.trainable = False

model.compile(optimizer=optimizer_ddi, loss='categorical_crossentropy', metrics = METRICS)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define class labels and their counts for insect bite dataset
class_labels = [0, 1, 2, 3, 4]  
class_counts = [169, 182, 173, 149, 206]

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.array(class_labels), y=np.repeat(class_labels, class_counts))
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# DDI dataset
history_ddi = model.fit(
    ddi_train_ds,
    validation_data=ddi_val_ds,
    epochs=10,  
    batch_size=64
)

# model.save_weights('model_weights.weights.h5')

for layer in model.layers:
    layer.trainable = False

# unfreeze last ten layers
for layer in model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=optimizer_ddi, loss='categorical_crossentropy', metrics = METRICS)

# ---------------------------------

model.pop()
model.add(Dense(5, activation='softmax'))
model.summary()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = METRICS)


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

class_names = ['ants', 'bed_bugs', 'chiggers', 'mosquitos', 'ticks']  # Adjust as needed

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Print class names with their accuracies
print("Per-Class Accuracy:")
for i, accuracy in enumerate(class_accuracy):
    print(f"{class_names[i]}: {accuracy:.2f}")

