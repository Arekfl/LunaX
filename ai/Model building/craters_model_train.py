from ultralytics import YOLO

def main():
    print("Rozpoczęcie treningu modelu")
    
    model = YOLO('yolov8m.pt') 

    results = model.train(
        data='craters_data.yaml',      
        epochs=50,                
        imgsz=640,                
        batch=16,                 
        project='crater_detection', 
        name='trening_1',  # zmieniać nazwe przy kolejnym treningu
        
        #zniekształcenia geometrii
        degrees=0.0,
        perspective=0.0,
        shear=0.0,
        fliplr=0.5,        
        flipud=0.5,               
        
        device=0                  
    )
    
    print("\nTrening modelu zakończony!")

if __name__ == '__main__':
    main()