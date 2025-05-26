import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

def train_animal_type():
    data_dir = "dataset/training_set"
    img_size = (150, 150)
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        verbose=1
    )
    
    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)
    print(f"Accuracy: {test_results[1]:.4f}")
    model.save("models/animal_type_model.h5")
    
    # Plot and save accuracy graph
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Animal Type Classification Accuracy')

    # Save the accuracy plot
    plt.savefig("accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    train_animal_type()
