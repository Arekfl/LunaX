import os
import requests
import random
from PIL import Image, ImageEnhance
from io import BytesIO
from ultralytics import YOLO

# zmieniać nazwe przy kolejnym treningu
MODEL_PATH = "runs/detect/crater_detection/trening_1/weights/best.pt"
OUTPUT_DIR = "craters_wyniki_testu"

def download_lunar_region(lon_min, lat_min, delta=2.0):
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
        "width": 1024,  
        "height": 1024,
        "format": "image/jpeg"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        if 'image' in response.headers.get('Content-Type', ''):
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            return img
        else:
            print("   Error: Serwer nie zwrócił obrazka.")
            return None
            
    except Exception as e:
        print(f"   Error: Połącznie {e}")
        return None

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Nie znaleziono modelu: {MODEL_PATH}")
        return

    #utworzenie folderu na wyniki
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # łądowanie modelu
    model = YOLO(MODEL_PATH)

    print("\nRozpoczęcie analizy")

    for i in range(1, 31):
        print(f"--- ANALIZA OBRAZU {i}/30 ---")
        
        # losowanie współrzędnych geograficznych
        lon_min = round(random.uniform(-170.0, 170.0), 2)
        lat_min = round(random.uniform(-70.0, 70.0), 2)
        
        print(f"Pobieranie regionu: Lon {lon_min}, Lat {lat_min}...")
        image = download_lunar_region(lon_min, lat_min)
        
        if image is None:
            print("Nie można pobrać orbazka, przechodzę do następnego.")
            continue
        
        # zapis czystego obrazka
        debug_path = os.path.join(OUTPUT_DIR, f"img_{i}_lon{lon_min}_lat{lat_min}_DEBUG.jpg")
        image.save(debug_path)

        # szukanie kraterów
        results = model.predict(source=image, conf=0.15, imgsz=1024, device=0, verbose=False)
        result = results[0]
        
        # tworzenie bounding boxów
        annotated_image = result.plot()
        final_img = Image.fromarray(annotated_image[..., ::-1]) 
        
        output_path = os.path.join(OUTPUT_DIR, f"img_{i}_lon{lon_min}_lat{lat_min}_WYNIK.jpg")
        final_img.save(output_path)
        
        if len(result.boxes) > 0:
            print(f"Znaleziono kraterów: {len(result.boxes)} -> Zapisano plik!")
        else:
            print("Model nie wykrył żadnych kraterów na tym zdjęciu.")
        print("-" * 40)

    print(f"\nAnaliza zakończona i zapisane w: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()