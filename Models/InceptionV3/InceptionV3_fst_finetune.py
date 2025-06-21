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

ddi_val_ds = ddi_val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2)))
ddi_train_ds = ddi_train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=2))) 

#--------------------------------------------------

base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape=(img_height, img_width, 3))

# Get the output from the 'mixed4' layer
mixed4_output = base_model.get_layer('mixed4').output

# Custom layers for DDI classification
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

base_model.trainable = True  		   # Unfreeze the base model for fine-tuning
for layer in base_model.layers[:249]:      # Keep earlier layers frozen
    layer.trainable = False

# Replace output layer for insect bite classification
x = ddi_model.layers[-3].output  # Output from the dense layer before softmax
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(5, activation='softmax')(x)  

final_model = models.Model(inputs=ddi_model.input, outputs=x)

val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=5)))

optimizer = Adadelta(learning_rate=1.0)

# Compile and train for insect bite classification
final_model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

print("Training on Insect Bite dataset...")
history_insect = final_model.fit(train_ds, validation_data=val_ds, epochs=50) 

#-----------------------------------------------------------------------------------

# Evaluate the final model
results = final_model.evaluate(test_ds)
print(f"Insect Bite Classification Test Loss: {results[0]}")
print(f"Insect Bite Classification Test Accuracy: {results[1]}")

#-----------------------------------------------------------------------------------

# The following code was used to identify what class bed bug bites were misclassified to. 

import numpy as np
import matplotlib.pyplot as plt

# Get predictions from the model
predictions = final_model.predict(test_ds, verbose=1)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

true_labels = []
images = []
for image_batch, label_batch in test_ds:
    true_labels.append(np.argmax(label_batch, axis=1))
    images.append(image_batch.numpy()) 

true_labels = np.concatenate(true_labels)
images = np.concatenate(images)

bed_bug_class_index = 1  # bed bug bites is the second class

# Find misclassified indices where the true label is "bed bug bites"
misclassified_indices = np.where((predicted_labels != true_labels) & (true_labels == bed_bug_class_index))[0]

# Find correctly classified indices where the true label is "bed bug bites"
correctly_classified_indices = np.where((predicted_labels == true_labels) & (true_labels == bed_bug_class_index))[0]

# Save misclassified bed bug bite images
num_images_to_show = 10
plt.figure(figsize=(12, 6))

for i, idx in enumerate(misclassified_indices[:num_images_to_show]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[idx].astype(np.uint8))  # Display the image
    plt.title(f"Pred: {predicted_labels[idx]}, True: {bed_bug_class_index}")
    plt.axis("off")

plt.savefig("misclassified_bed_bug_bites.png")
plt.close() 

# Save correctly classified bed bug bite images
plt.figure(figsize=(12, 6))

for i, idx in enumerate(correctly_classified_indices[:num_images_to_show]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[idx].astype(np.uint8))  # Display the image
    plt.title(f"Pred: {predicted_labels[idx]}, True: {bed_bug_class_index}")
    plt.axis("off")

plt.savefig("correctly_classified_bed_bug_bites.png")
plt.close()
