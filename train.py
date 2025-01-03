from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt") #pretrained model

    # Train
    results = model.train(
        data=r'C:\Debin\Kuliah\Semester7\VisiKomputer\Final\Traffic-Sign-in-Indonesia-Detection-3\data.yaml',
        epochs=20,
        imgsz=640
    )