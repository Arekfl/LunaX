import os

folder = "lunar_background"
for file in os.listdir(folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        # Tworzy nazwę pliku .txt na podstawie nazwy zdjęcia
        txt_name = os.path.splitext(file)[0] + ".txt"
        txt_path = os.path.join(folder, txt_name)
        # Tworzy pusty plik
        open(txt_path, 'a').close()

print("Gotowe. Puste pliki .txt zostały utworzone.")