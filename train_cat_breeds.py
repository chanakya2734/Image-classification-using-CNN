import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    return model

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def parse_data(x, y):
    x = x.decode()
    num_class = 55  # Number of cat breeds
    size = 224

    image = read_image(x, size)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)

    return image, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((224, 224, 3))
    y.set_shape((55,))
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

if __name__ == "__main__":
    path = r"C:\Users\chanakya\OneDrive\Desktop\cat_dog_classification\dataset\breed_dataset\cat_breeds"
    train_path = os.path.join(path, "train", "*")
    test_path = os.path.join(path, "test", "*")
    labels_path = os.path.join(path, "cat_labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Cat Breeds:", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}

    image_paths = glob(train_path)
    labels = []
    clean_image_paths = []

    for image_path in image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        row = labels_df[labels_df['id'].str.lower() == (image_id + ".jpg").lower()]
        if not row.empty:
            breed_name = row["breed"].values[0]
            breed_idx = breed2id[breed_name]
            labels.append(breed_idx)
            clean_image_paths.append(image_path)
        else:
            print(f"Warning: Image ID '{image_id}' not found in cat_labels.csv")

    print("Loaded CSV head:\n", labels_df.head())
    print("CSV ID column type:", labels_df['id'].dtype)
    print(f"Total matched images: {len(clean_image_paths)}")

    # Optional: limit for faster testing
    clean_image_paths = clean_image_paths[:1000]
    labels = labels[:1000]

    # Splitting the dataset
    train_x, valid_x, train_y, valid_y = train_test_split(clean_image_paths, labels, test_size=0.2, random_state=42)

    # Parameters
    size = 224
    num_classes = 55
    lr = 1e-4
    batch = 16
    epochs = 10

    # Model
    model = build_model(size, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])

    # Dataset
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    # Training
    callbacks = [
        ModelCheckpoint("models/cat_breed_model.h5", verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
    ]

    train_steps = (len(train_x) // batch) + 1
    valid_steps = (len(valid_x) // batch) + 1

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    # Plot accuracy graph and save as accuracy.png
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Cat Breed Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('cat_breed_accuracy.png')
    plt.close()
