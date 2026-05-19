import os
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# 1. KONFIGURACJA
MODEL_PATH = "runs/detect/lunar_project/trening_v1-2/weights/best.pt" 
OUTPUT_IMAGE = "test_result.jpg"

def download_lunar_region():
    print("Łączenie z serwerem WMS USGS...")
    
    # 1. ZWIĘKSZAMY OBSZAR (delta = 2.0 stopnie zamiast 0.5)
    lon_min = -58.0
    lat_min = 13.0
    delta = 2.0 
    lon_max = lon_min + delta
    lat_max = lat_min + delta
    
    url = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
    
    params = {
        "map": "/maps/earth/moon_simp_cyl.map", 
        "request": "GetMap",
        "service": "WMS",
        "version": "1.1.1",
        "layers": "LROC_WAC",
        "styles": "",
        "srs": "EPSG:4326",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        # 2. ZWIĘKSZAMY ROZDZIELCZOŚĆ (aby zachować detale)
        "width": 1280,  
        "height": 1280,
        "format": "image/jpeg"
    }
    # ... reszta funkcji bez zmian ...
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        if 'image' in response.headers.get('Content-Type', ''):
            img = Image.open(BytesIO(response.content))
            # YOLO preferuje RGB, więc zmieniamy konwersję z "L" na "RGB"
            img = img.convert("RGB")
            print(f"SUKCES: Pobrano wycinek Księżyca (Lon: {lon_min} do {lon_max}, Lat: {lat_min} do {lat_max})")
            return img
        else:
            print("BŁĄD: Serwer nie zwrócił obrazka.")
            return None
            
    except Exception as e:
        print(f"BŁĄD POŁĄCZENIA: {e}")
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono modelu: {MODEL_PATH}")
        return

    # 1. Pobranie obrazka
    image = download_lunar_region()
    if image is None:
        return

    # 2. Załadowanie modelu
    print("Ładowanie modelu YOLOv11...")
    model = YOLO(MODEL_PATH)

    # 3. Szukanie obiektów (conf=0.25 odrzuca bardzo słabe trafienia)
    print("Rozpoczynam analizę obrazu...")
    results = model.predict(source=image, conf=0.25, imgsz=1280, device=0)
    result = results[0]
    
    # 4. Rysowanie ramek i zapis
    annotated_image = result.plot()
    final_img = Image.fromarray(annotated_image[..., ::-1]) 
    final_img.save(OUTPUT_IMAGE)
    
    print(f"\nGotowe! Zobacz plik: {OUTPUT_IMAGE}")
    
    if len(result.boxes) > 0:
        print(f"Znaleziono podejrzanych miejsc: {len(result.boxes)}")
    else:
        print("Model nie wykrył żadnych jam na tym zdjęciu.")

if __name__ == "__main__":
    main()