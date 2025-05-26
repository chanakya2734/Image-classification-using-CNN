import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    return image.astype(np.float32) / 255.0

if __name__ == "__main__":
    # Paths
    base_path = "C:/Users/chanakya/OneDrive/Desktop/cat_dog_classification/dataset/breed_dataset/dog_breeds"
    labels_path = os.path.join(base_path, "labels.csv")
    image_folder = os.path.join(base_path, "train")
    model_path = "C:/Users/chanakya/OneDrive/Desktop/cat_dog_classification/models/dog_breed_model.h5"

    # Load labels and build mappings
    labels_df = pd.read_csv(labels_path)
    breed_list = labels_df["breed"].unique()
    breed2id = {b: i for i, b in enumerate(breed_list)}
    id2breed = {i: b for i, b in enumerate(breed_list)}

    # Filter images that exist
    image_paths, labels = [], []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(image_folder, f"{row['id']}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(breed2id[row["breed"]])

    # Optional: Limit for quick testing
    image_paths = image_paths[:1000]
    labels = labels[:1000]

    # Split data (stratify ensures balanced classes)
    train_x, valid_x, train_y, valid_y = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Output directory
    os.makedirs("save", exist_ok=True)

    # Read and predict in batch
    selected_paths = valid_x[:10]
    selected_labels = valid_y[:10]

    batch_images = np.array([read_image(path, 224) for path in selected_paths])
    predictions = model.predict(batch_images)

    for i, (img_path, true_label, pred) in enumerate(zip(selected_paths, selected_labels, predictions)):
        pred_idx = np.argmax(pred)
        pred_breed = id2breed[pred_idx]
        true_breed = id2breed[true_label]
        confidence = pred[pred_idx] * 100

        img = cv2.imread(img_path)
        img = cv2.putText(img, f"Pred: {pred_breed} ({confidence:.2f}%)", (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        img = cv2.putText(img, f"True: {true_breed}", (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(f"dog_save/valid_{i}.png", img)

    print("✅ Predictions saved to 'dog_save/' folder.")
