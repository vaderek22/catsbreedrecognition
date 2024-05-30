import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Funkcja do przetwarzania zdjęcia przed przewidywaniem
def preprocess_image_for_prediction(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Funkcja obsługująca przycisk
def check_image():
    image_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                           filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    target_size = (224, 224)  # Możesz ustawić rozmiar na taki, jaki jest wymagany przez model

    # Przetwarzanie zdjęcia
    img = preprocess_image_for_prediction(image_path, target_size)

    # Przewidywanie
    prediction = model.predict(img)
    breeds = ['Sfinks', 'Maine Coon', 'Syjamski', 'Brytyjski']
    max_prob = np.max(prediction)
    if max_prob < 0.5:
        result_label.config(text="Na zdjęciu nie ma kota.")
        probabilities_label.config(text="")
        info_label.config(text="")
    else:
        predicted_breed = breeds[np.argmax(prediction)]
        probabilities = [f'{breeds[i]}: {round(prob * 100, 2)}%' for i, prob in enumerate(prediction[0])]

        # Wyświetlenie wyników
        result_label.config(text=f'Na zdjęciu znajduje się kot rasy {predicted_breed}')
        probabilities_label.config(text='\n'.join(probabilities))

        # Wyświetlenie ciekawostek
        if predicted_breed == 'Maine Coon':
            info_label.config(
                text="Maine Coon: To jedna z największych ras domowych kotów na świecie, a ich charakterystyczną cechą są puszyste ogony i gęste, wodoodporne futra.",
                font=('calibri', 10))
        elif predicted_breed == 'Sfinks':
            info_label.config(
                text="Sfinks: Pomimo swojego bezwłosego wyglądu, sfinksy posiadają delikatną warstwę meszku, co daje im aksamitne w dotyku ciało.",
                font=('calibri', 10))
        elif predicted_breed == 'Syjamski':
            info_label.config(
                text="Syjamski: To rasa kotów o długiej historii, znana z charakterystycznego wyglądu, w tym niebieskich oczu i kolorowych punktów na sierści.",
                font=('calibri', 10))
        elif predicted_breed == 'Brytyjski':
            info_label.config(
                text="Brytyjski: Ten kot charakteryzuje się krępej budowie ciała, okrągłą głową i krótkim, gęstym futrem.",
                font=('calibri', 10))

    # Wyświetlenie podglądu załadowanego zdjęcia
    load = Image.open(image_path)
    load.thumbnail((400, 400))
    render = ImageTk.PhotoImage(load)
    img_label.config(image=render)
    img_label.image = render

# Wczytanie wytrenowanego modelu
model = load_model('cats_breed_model_efficientnet.keras')

# Interfejs użytkownika
root = tk.Tk()
root.title("Rozpoznawanie rasy kota")
root.geometry("850x850")

# Stylizacja aplikacji
style = ttk.Style(root)
style.configure('TButton', font=('calibri', 12, 'bold'), foreground='black', padding=20)
style.configure('TLabel', font=('calibri', 14), foreground='black')
style.configure('TFrame', background='white')

# Przycisk wyboru pliku
file_button = ttk.Button(root, text="Wybierz plik", command=check_image)
file_button.pack(pady=20)

# Etykieta na podgląd zdjęcia
img_label = ttk.Label(root)
img_label.pack(pady=20)

# Etykieta na wynik
result_label = ttk.Label(root, text="")
result_label.pack(pady=20)

# Etykieta na prawdopodobieństwa
probabilities_label = ttk.Label(root, text="", font=('calibri', 9))
probabilities_label.pack(pady=20)

# Etykieta na informacje o rasie kota
info_label = ttk.Label(root, text="", font=('calibri', 12))
info_label.pack(pady=20)

root.mainloop()
