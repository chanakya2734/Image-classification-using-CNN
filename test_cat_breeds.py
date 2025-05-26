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
    image = image / 255.0
    image = image.astype(np.float32)
    return image

if __name__ == "__main__":
    path = "C:/Users/chanakya/OneDrive/Desktop/cat_dog_classification/dataset/breed_dataset/cat_breeds"
    train_path = os.path.join(path, "train", "*")
    test_path = os.path.join(path, "test", "*")
    labels_path = os.path.join(path, "cat_labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Cat Breeds: ", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}
    id2breed = {i: name for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []
    clean_ids = []  # Keep only those image paths for which label is found

    for image_path in ids:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        breed_row = labels_df[labels_df.id.str.lower() == (image_id + ".jpg").lower()]
        if not breed_row.empty:
            breed_name = breed_row["breed"].values[0]
            breed_idx = breed2id[breed_name]
            labels.append(breed_idx)
            clean_ids.append(image_path)
        else:
            print(f"Warning: ID '{image_id}' not found in cat_labels.csv")

    # Optional: limit for testing
    clean_ids = clean_ids[:1000]
    labels = labels[:1000]

    ## Splitting dataset
    train_x, valid_x = train_test_split(clean_ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    ## ✅ Load pretrained CAT breed model
    model_path = "C:/Users/chanakya/OneDrive/Desktop/cat_dog_classification/models/cat_breed_model.h5"
    model = tf.keras.models.load_model(model_path)

    ## Output folder
    os.makedirs("cat_save", exist_ok=True)

    for i, image_path in tqdm(enumerate(valid_x[:10]), total=10):
        image = read_image(image_path, 224)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)[0]
        label_idx = np.argmax(pred)
        breed_name = id2breed[label_idx]
        confidence = pred[label_idx] * 100

        ori_breed = id2breed[valid_y[i]]
        ori_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        ori_image = cv2.putText(ori_image, f"Pred: {breed_name} ({confidence:.2f}%)", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        ori_image = cv2.putText(ori_image, f"True: {ori_breed}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite(f"cat_save/valid_{i}.png", ori_image)

    print("✅ Cat breed test images with predictions saved in 'cat_save/' folder.")
