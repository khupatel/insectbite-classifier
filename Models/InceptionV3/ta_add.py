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

#--------------------------------------------------
# DDI dataset
ddi_data_dir = pathlib.Path(r'/home/kvp5640/THESIS/ddi_dataset')

# train dataset DDI 
ddi_train_ds = tf.keras.utils.image_dataset_from_directory(
    ddi_data_dir / 'train',  
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# validation dataset DDI 
ddi_val_ds = tf.keras.utils.image_dataset_from_directory(
    ddi_data_dir / 'validation',  
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

#--------------------------------------------------

# Model A

ddi_val_ds = ddi_val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2)))
ddi_train_ds = ddi_train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2))) 

base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(img_height, img_width, 3))

# Get the output from the 'mixed4' layer
mixed4_output = base_model.get_layer('mixed4').output

# Add custom layers for DDI classification
x = layers.GlobalAveragePooling2D()(mixed4_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
ddi_output = layers.Dense(2, activation='softmax')(x)

ddi_model = models.Model(inputs=base_model.input, outputs=ddi_output)

# Compile and train on DDI dataset
ddi_model.compile(optimizer=Adadelta(learning_rate=1.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("Training on DDI dataset...")
history_ddi = ddi_model.fit(ddi_train_ds, validation_data=ddi_val_ds, epochs=20)

# Save weights after DDI training
ddi_model.save_weights('ddi.weights.h5')

#------------------
# Model B

val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))

base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(img_height, img_width, 3))

# Get the output from the 'mixed4' layer
mixed4_output = base_model.get_layer('mixed4').output

# Add custom layers for DDI classification
x = layers.GlobalAveragePooling2D()(mixed4_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
insect_output = layers.Dense(5, activation='softmax')(x)

insect_model = models.Model(inputs=base_model.input, outputs=insect_output)

optimizer = Adadelta(learning_rate=1.0)

# Compile and train for insect bite classification
insect_model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

print("Training on Insect Bite dataset...")
history_insect = insect_model.fit(train_ds, validation_data=val_ds, epochs=50)

#---------------

# Load DDI-trained weights
ddi_model.load_weights('ddi.weights.h5')

# Transfer all but the output layer
#for layer_insect, layer_ddi in zip(insect_model.layers[:-1], ddi_model.layers[:-1]):
#    layer_insect.set_weights(layer_ddi.get_weights())


# Perform Task Arithmetic: Add all but the output layer
for layer_insect, layer_ddi in zip(insect_model.layers[:-1], ddi_model.layers[:-1]):
    new_weights = [(w_insect + w_ddi) for w_insect, w_ddi in zip(layer_insect.get_weights(), layer_ddi.get_weights())]
    layer_insect.set_weights(new_weights)

print("Transferred all but the output layer from DDI model to Insect model.")

for layer in ddi_model.layers[-10:]:  # Fine-tune the last 10 layers
    layer.trainable = True

insect_model.compile(optimizer=Adadelta(learning_rate=1.0),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

print("Fine-tuning Insect Model with DDI knowledge...")
history_insect_finetune = insect_model.fit(train_ds, validation_data=val_ds, epochs=20)

# Save the fine-tuned model
insect_model.save_weights('insect_finetune.weights.h5')

#-----------------------------------------------------------------------------------
# Evaluate the final model

results = insect_model.evaluate(test_ds)
print(f"Insect Bite Classification Test Loss: {results[0]}")
print(f"Insect Bite Classification Test Accuracy: {results[1]}")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on the test dataset
y_pred = insect_model.predict(test_ds, batch_size=batch_size) 
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








