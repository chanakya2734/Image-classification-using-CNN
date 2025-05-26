import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

try:
    animal_model = load_model("models/animal_type_model.h5")
    cat_breed_model = load_model("models/cat_breed_model.h5")
    dog_breed_model = load_model("models/dog_breed_model.h5")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model(s): {str(e)}")
    exit()

animal_input_size = animal_model.input_shape[1]

cat_breeds = [ 'Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair',
    'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese',
    'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex',
    'Devon Rex', 'Dilute Calico', 'Dilute Tortoise shell', 'Domestic Long Hair', 'Domestic Medium Hair',
    'Domestic Short Hair', 'Egyptian Mau', 'Exotic Short Hair', 'Extra Toes cat - Hemingway Polydactyl',
    'Havana', 'Himalayan', 'Japanese Bobtail', 'Korat', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung',
    'Norweigan Forest Cat', 'Ocicat', 'Oriental Short Hair', 'Persian', 'Ragdoll', 'Russian Blue',
    'Scottish Fold', 'Siamese', 'Siberian', 'Snowshoe', 'Sphynx-Hairless Cat', 'Tabby', 'Tiger',
    'Tonkinese', 'Torbie', 'Tortoise Shell', 'Turkish Angora', 'Turkish Van', 'Tuxedo'
]

dog_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
    'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier',
    'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer',
    'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
    'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
    'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound',
    'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
    'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
    'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog',
    'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel',
    'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg',
    'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher',
    'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke',
    'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
    'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
    'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier',
    'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
    'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
    'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
    'wire-haired_fox_terrier', 'yorkshire_terrier'
]

top = tk.Tk()
top.geometry("600x600")
top.title("Image Classification")
top.configure(background="#CDCDCD")

heading = tk.Label(top, text="Upload Your Image for Classification", pady=20, font=("arial", 20, "bold"), bg="#CDCDCD", fg="#364156")
heading.pack()

sign_image = tk.Label(top)
sign_image.pack(pady=10)

label_prediction = tk.Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
label_prediction.pack()

label_accuracy = tk.Label(top, background="#CDCDCD", font=("arial", 12))
label_accuracy.pack()

label_breed = tk.Label(top, background="#CDCDCD", font=("arial", 12))
label_breed.pack()

classify_btn = tk.Button(top, text="Classify", padx=10, pady=5, bg="#364156", fg="white", font=("arial", 10, "bold"))
classify_btn.pack(pady=5)
classify_btn.pack_forget()

def preprocess_image(path, target_size):
    try:
        img = Image.open(path).convert("RGB").resize((target_size, target_size))
        img = np.expand_dims(np.array(img) / 255.0, axis=0)
        return img
    except Exception as e:
        messagebox.showerror("Error", f"Image processing failed: {str(e)}")
        return None

def classify(file_path):
    try:
        input_img_animal = preprocess_image(file_path, animal_input_size)
        if input_img_animal is None:
            return
        animal_pred = animal_model.predict(input_img_animal)[0][0]

        if animal_pred > 0.5:
            label = "Dog"
            animal_confidence = round(animal_pred * 100, 2)
            breed_model = dog_breed_model
            breed_list = dog_breeds
        else:
            label = "Cat"
            animal_confidence = round((1 - animal_pred) * 100, 2)
            breed_model = cat_breed_model
            breed_list = cat_breeds

        breed_input_size = breed_model.input_shape[1]
        input_img_breed = preprocess_image(file_path, breed_input_size)
        if input_img_breed is None:
            return
        breed_pred = breed_model.predict(input_img_breed)[0]
        breed_index = np.argmax(breed_pred)
        breed_name = breed_list[breed_index] if breed_index < len(breed_list) else "Unknown"
        breed_confidence = round(np.max(breed_pred) * 100, 2)

        label_prediction.config(text=f"Animal: {label}")
        label_accuracy.config(text=f"Classifier Accuracy: {animal_confidence}%")
        label_breed.config(text=f"Predicted Breed: {breed_name} ({breed_confidence}%)")

    except Exception as e:
        messagebox.showerror("Error", f"Classification failed: {str(e)}")

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    try:
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((250, 250))
        imgtk = ImageTk.PhotoImage(image)
        sign_image.configure(image=imgtk)
        sign_image.image = imgtk

        label_prediction.config(text="")
        label_accuracy.config(text="")
        label_breed.config(text="")

        classify_btn.config(command=lambda: classify(file_path))
        classify_btn.pack()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {str(e)}")

upload_btn = tk.Button(top, text="Upload Image", command=upload_image, padx=10, pady=5,
                       background="#364156", foreground="white", font=("arial", 10, "bold"))
upload_btn.pack(pady=20)

top.mainloop()