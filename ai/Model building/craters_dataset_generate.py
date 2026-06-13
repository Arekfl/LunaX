import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

CSV_PATH = "craters_raw_data.csv"
OUTPUT_DIR = "craters_images_dataset"
NUM_IMAGES = 9999
IMAGE_SIZE = 640
BBOX_SIZE_DEG = 3.0  

COL_LAT = "Latitude"         
COL_LON = "Longitude"         
COL_DIAM = "Diameter"   
COL_AGE = "Age" 

KM_PER_DEGREE = 30.3  

# powiekszenie bounding boxa
MARGIN_FACTOR = 1.25

# ściąganie zdjec 
def download_wms(lon_min, lat_min, lon_max, lat_max):
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
        "width": IMAGE_SIZE,  
        "height": IMAGE_SIZE,
        "format": "image/jpeg"
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"   Błąd: Kod {response.status_code}, Typ danych: {response.headers.get('Content-Type', '')}")
            print(f"   Treść błędu z USGS: {response.text[:250]}")
    except Exception as e:
        print(f"błąc połączennie: {e}")
    return None

def get_class(age_val):
    try:
        val = int(float(age_val))
        if 1 <= val <= 6:
            return val - 1
    except:
        pass
    return -1

def main():
    print("Wczytywanie bazy danych.")
    try:
        df = pd.read_csv(CSV_PATH, low_memory=False, sep=None, engine='python')
    except Exception as e:
        df = pd.read_csv(CSV_PATH, low_memory=False)
        
    print(f"   Wczytano surowych wierszy: {len(df)}")

# czyszczenie danych
    for col in [COL_LAT, COL_LON, COL_DIAM]:
        df[col] = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[COL_LAT, COL_LON, COL_DIAM])

    if COL_AGE not in df.columns:
        print(f"\nBłąd, nie znaleziono kolumny '{COL_AGE}'.")
        print(f"Dostępne kolumny to: {df.columns.tolist()[:10]}")
        return
        
    df = df.dropna(subset=[COL_AGE])
    
    # Mapowanie na 6 klas wiekowych
    df['YOLO_Class'] = df[COL_AGE].apply(get_class)
    df = df[df['YOLO_Class'] != -1]
    
    if len(df) == 0:
        print("\nBłąd: Po wyczyszczeniu danych nie został żaden krater! Upewnij się, że kolumna z wiekiem ma wartości od 1 do 6.")
        return

    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    
    print(f"\n2. Baza gotowa i wyczyszczona. Poprawnych kraterów do nauki: {len(df)}")
    print("3. Rozpoczynam pobieranie map z serwerów NASA/USGS i generowanie etykiet...")

    # próbka kraterów
    sample_craters = df.sample(n=min(NUM_IMAGES, len(df)), random_state=42)
    
    success_count = 0
    for idx, row in sample_craters.iterrows():
        c_lat = row[COL_LAT]
        c_lon = row[COL_LON]
        
        lon_min = c_lon - (BBOX_SIZE_DEG / 2)
        lon_max = c_lon + (BBOX_SIZE_DEG / 2)
        lat_min = c_lat - (BBOX_SIZE_DEG / 2)
        lat_max = c_lat + (BBOX_SIZE_DEG / 2)
        
        # inne kratery w tym kadrze
        craters_in_view = df[
            (df[COL_LAT] >= lat_min) & (df[COL_LAT] <= lat_max) &
            (df[COL_LON] >= lon_min) & (df[COL_LON] <= lon_max)
        ]
        
        img = download_wms(lon_min, lat_min, lon_max, lat_max)
        if img is None:
            continue
            
        img_filename = f"crater_tile_{success_count}.jpg"
        txt_filename = f"crater_tile_{success_count}.txt"
        
        yolo_labels = []
        for _, crater in craters_in_view.iterrows():
            # przeliczanie rozmiaru z marginesem
            diam_deg = (crater[COL_DIAM] * MARGIN_FACTOR) / KM_PER_DEGREE
            
            # obliczanie pozycji
            x_c = (crater[COL_LON] - lon_min) / BBOX_SIZE_DEG
            y_c = (lat_max - crater[COL_LAT]) / BBOX_SIZE_DEG
            w = diam_deg / BBOX_SIZE_DEG
            h = diam_deg / BBOX_SIZE_DEG
            
            # clamping
            x_min = max(0.0, x_c - w/2); x_max = min(1.0, x_c + w/2)
            y_min = max(0.0, y_c - h/2); y_max = min(1.0, y_c + h/2)
            
            # na format YOLO
            new_x = (x_min + x_max) / 2
            new_y = (y_min + y_max) / 2
            new_w = x_max - x_min
            new_h = y_max - y_min
            
            if new_w > 0.01 and new_h > 0.01:
                yolo_labels.append(f"{int(crater['YOLO_Class'])} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")
        
        if yolo_labels:
            img.save(os.path.join(OUTPUT_DIR, "images", "train", img_filename))
            with open(os.path.join(OUTPUT_DIR, "labels", "train", txt_filename), 'w') as f:
                f.write('\n'.join(yolo_labels))
            success_count += 1
            print(f"   [{success_count}/{NUM_IMAGES}] Zapisano zdjęcie. Kraterów na zdjęciu: {len(yolo_labels)}")

    print(f"\nZakończono generowanie datasetu: '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()