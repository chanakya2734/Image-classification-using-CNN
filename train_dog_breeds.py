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

AUTOTUNE = tf.data.experimental.AUTOTUNE

def build_model(input_size, num_classes):
    inputs = Input((input_size, input_size, 3))
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def read_image(path, size):
    img = cv2.imread(path.decode(), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0

def parse_data(path, label, size=224, num_classes=120):
    img = read_image(path, size)
    one_hot = np.eye(num_classes)[label]
    return img, one_hot

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.float32])
    x.set_shape([224, 224, 3])
    y.set_shape([120])
    return x, y

def create_dataset(x, y, batch=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.map(tf_parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch).prefetch(AUTOTUNE)
    return dataset

if __name__ == "__main__":
    path = "C:/Users/chanakya/OneDrive/Desktop/cat_dog_classification/dataset/breed_dataset/dog_breeds"
    train_dir = os.path.join(path, "train")
    labels_df = pd.read_csv(os.path.join(path, "labels.csv"))

    breed_list = labels_df["breed"].unique()
    breed2id = {b: i for i, b in enumerate(breed_list)}
    labels_df["label"] = labels_df["breed"].map(breed2id)

    # Map images to paths and labels
    image_paths = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(train_dir, row["id"] + ".jpg")
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(row["label"])

    # Optional: Limit data for quick testing
    image_paths = image_paths[:1000]
    labels = labels[:1000]

    # Split data
    train_x, valid_x, train_y, valid_y = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    # Parameters
    size = 224
    num_classes = len(breed_list)
    batch_size = 16
    epochs = 10
    lr = 1e-4

    # Model
    model = build_model(size, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["accuracy"])

    # Dataset
    train_ds = create_dataset(train_x, train_y, batch=batch_size, shuffle=True)
    valid_ds = create_dataset(valid_x, valid_y, batch=batch_size, shuffle=False)

    steps_per_epoch = len(train_x) // batch_size
    validation_steps = len(valid_x) // batch_size

    # Callbacks
    callbacks = [
        ModelCheckpoint("models/dog_breed_model.h5", save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.1, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("dog_breed_accuracy.png")
    plt.close()
