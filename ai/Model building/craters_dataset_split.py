import os
import random
import shutil

SRC_IMG_DIR = "craters_dataset_relabel/images"
SRC_LBL_DIR = "craters_dataset_relabel/labels"
DEST_DIR = "craters_training_dataset_yolo"

# train, val, test

SPLIT_RATIO = (0.8, 0.1, 0.1) 

def setup_directories(base_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

def main():
    print("Skanowanie plików.")
    images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith('.jpg')]
    valid_pairs = []
    
    for img in images:
        txt = img.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(SRC_LBL_DIR, txt)):
            valid_pairs.append((img, txt))
            
    print(f"Znaleziono {len(valid_pairs)} poprawnych par zdjęcie + etykieta.")
    
    # mieszanie danych
    random.seed(42)
    random.shuffle(valid_pairs)
    
    # indeksy podziału
    total = len(valid_pairs)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = train_end + int(total * SPLIT_RATIO[1])
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    print("Kopiowanie i podział plików.")
    setup_directories(DEST_DIR)
    
    for split_name, files in splits.items():
        for img, txt in files:
            shutil.copy(
                os.path.join(SRC_IMG_DIR, img),
                os.path.join(DEST_DIR, 'images', split_name, img)
            )
            shutil.copy(
                os.path.join(SRC_LBL_DIR, txt),
                os.path.join(DEST_DIR, 'labels', split_name, txt)
            )
        print(f"  - Zbiór '{split_name}': {len(files)} plików.")
        
    print(f"\nKoniec, zbiór to treningu utworzony: '{DEST_DIR}'.")

if __name__ == '__main__':
    main()