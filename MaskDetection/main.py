## Importing libraries
import pandas as pd
import numpy as np
import cv2
import os
import glob
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import seaborn as sns

from tensorflow.python.keras import Sequential, models
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

path = "C:/Users/Tomek/PycharmProjects/MaskDetection/Trained_datasets/Face Mask Dataset/"

dataset = {
    "image_path": [],
    "mask_status": [],
    "where": []
}

for where in os.listdir(path):
    for status in os.listdir(path + "/" + where):
        for image in glob.glob(path + where + "/" + status + "/" + "*.png"):
            dataset["image_path"].append(image)
            dataset["mask_status"].append(status)
            dataset["where"].append(where)

dataset = pd.DataFrame(dataset)
dataset.head()

## Wybranie losowego zdjecia do algorytmu
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

## Wybor zdjecia z folderu ze zdjeciami
img = cv2.imread("C:/Users/Tomek/PycharmProjects/MaskDetection/Dataset/images/maksssksksss352.png")

## Konwersja na czarno-bialy na bazie algorytmu
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

## Zwraca koordynaty w formie numpy wszystkich wykrytych twarzy
detected_face = face_model.detectMultiScale(img)

## Konwersja powerotna z czarno-bialego zdjecia na kolorowe
output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

## Obrys wykrytych twarzy
for (x, y, w, h) in detected_face:
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 200), 2)

## Pokazanie obrazka z wykrytymi twarzami
plt.figure(figsize=(15, 15))
plt.imshow(output_img)
plt.show()


## Wypisanie ile jest obrazkow z maskami i bez
print(f"With Mask:", dataset.value_counts("mask_status")[0])
print(f"Without Mask:", dataset.value_counts("mask_status")[1])

## Utworzenie wykresu
sns.countplot(x=dataset["mask_status"])

plt.figure(figsize=(15, 10))

for i in range(9):
    random = np.random.randint(1, len(dataset))
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.imread(dataset.loc[random, "image_path"]))
    plt.title(dataset.loc[random, "mask_status"], size=15)
    plt.xticks([])
    plt.yticks([])

plt.show()

# Podzial na zbiory
train_df = dataset[dataset["where"] == "Train"]
test_df = dataset[dataset["where"] == "Test"]
valid_df = dataset[dataset["where"] == "Validation"]

print(train_df.head(10))

## Randomizowanie datasetu
train_df = train_df.sample(frac = 1)
test_df = test_df.sample(frac = 1)
valid_df = valid_df.sample(frac = 1)

print("\n After Shuffling \n")
print(train_df.head(10))

data = []
image_size = 150

## Wyswietlanie wytrenowanych zdjec
for i in range(len(train_df)):
    ## konersja na czarno-biale
    img_array = cv2.imread(train_df["image_path"][i], cv2.IMREAD_GRAYSCALE)

    new_image_array = cv2.resize(img_array, (image_size, image_size))

    ## Wpisanie do odpowiedniej tablicy
    if train_df["mask_status"][i] == "WithMask":
        data.append([new_image_array, 1])
    else:
        data.append([new_image_array, 0])

data = np.array(data)

np.random.shuffle(data)

fig, ax = plt.subplots(2, 3, figsize=(10, 10))

## Wyswietlanie 9 zdjec ze zbioru treningowego
for row in range(2):
    for col in range(3):
        image_index = row * 100 + col

        ax[row, col].axis("off")
        ax[row, col].imshow(data[image_index][0], cmap="gray")

        if data[image_index][1] == 0:
            ax[row, col].set_title("Without Mask")
        else:
            ax[row, col].set_title("With Mask")

plt.show()

X = []
y = []

##
for image in data:
    X.append(image[0])
    y.append(image[1])

## Konwersja na numpy (potrzebne dp TensorFlowa)
X = np.array(X)
y = np.array(y)

## Normalizacja danych
X = X/255

## Inicjalizacja procesu trenowania, okreslenie funkcji aktywnacji
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

## kompilacja
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

## Reshaping training set to match Conc2D
X_train = X_train.reshape(len(X_train), X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(len(X_val), X_val.shape[1], X_val.shape[2], 1)

history = model.fit(X_train, y_train, epochs=5, batch_size = 32)

model.summary()

model.evaluate(X_val, y_val)

## Wykonanie predykcji
prediction = model.predict_classes(X_val)

print("raporty")
print(classification_report(y_val, prediction))
print(confusion_matrix(y_val, prediction))
