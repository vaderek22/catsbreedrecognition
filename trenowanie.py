import cv2
import os
import numpy as np
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split

# Ścieżki do folderów ze zdjęciami
sfinks_path = './sfinks'
mainecoon_path = './mainecoon'
syjamski_path = './syjamski'
brytyjski_path = "./brytyjski"

# Ustal rozmiar, na jaki chcesz przeskalować zdjęcia
target_size = (224, 224)

# Funkcja do przeskalowania zdjęć
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    return img

# Przygotowanie danych
data = []
labels = []

for img_file in os.listdir(sfinks_path):
    img_path = os.path.join(sfinks_path, img_file)
    img = preprocess_image(img_path, target_size)
    data.append(img)
    labels.append(0)  # 0 dla sfinksa

for img_file in os.listdir(mainecoon_path):
    img_path = os.path.join(mainecoon_path, img_file)
    img = preprocess_image(img_path, target_size)
    data.append(img)
    labels.append(1)  # 1 dla maine coon

for img_file in os.listdir(syjamski_path):
    img_path = os.path.join(syjamski_path, img_file)
    img = preprocess_image(img_path, target_size)
    data.append(img)
    labels.append(2)  # 2 dla kotów syjamskich

for img_file in os.listdir(brytyjski_path):
    img_path = os.path.join(brytyjski_path, img_file)
    img = preprocess_image(img_path, target_size)
    data.append(img)
    labels.append(3)  # 3 dla kotów syjamskich

data = np.array(data)
labels = np.array(labels)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Tworzenie modelu EfficientNetB0 z regularyzacją
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # Dodanie warstwy Dropout
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Zapisz model do pliku w formacie Keras
model.save('cats_breed_model_efficientnet.keras')

# Możesz teraz kontynuować dalsze dostosowywanie modelu do swoich potrzeb.
