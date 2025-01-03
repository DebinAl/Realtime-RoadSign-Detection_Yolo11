import os
import torch
import numpy as np
import cv2
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH = r"Model\Road_Sign_YOLO.pt"

# List of class names
CLASS_NAMES = [
    'Balai Pertolongan Pertama', 'Banyak Anak-Anak', 'Dilarang Belok Kanan', 'Dilarang Berhenti',
    'Berhenti', 'Dilarang Masuk', 'Dilarang Mendahului', 'Dilarang Parkir',
    'Dilarang Putar Balik', 'Gereja', 'Hati-Hati', 'Jalur Penyebrangan', 'Lampu Lalu Lintas',
    'Larangan Kecepatan - 30km-jam', 'Larangan Kecepatan - 40km-jam', 'Larangan Kendaraan MST - 10 Ton',
    'Masjid', 'Pemberhentian Bus', 'Perintah Ikuti Bundaran', 'Perintah Jalur Sepeda', 'Perintah Lajur Kiri',
    'Perintah Pilih Satu Jalur', 'Persimpangan 3 Prioritas', 'Persimpangan 3 Sisi Kanan Prioritas',
    'Persimpangan 3 Sisi Kiri Prioritas', 'Persimpangan Empat', 'Putar Balik', 'Rumah Sakit', 'SPBU', 'Tempat Parkir'
]

CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(CLASS_NAMES)}

class ObjectDetection:

    def __init__(self, source):
        self.source = source
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()

        self.class_confidences = defaultdict(list)

    def load_model(self):
        model = YOLO(MODEL_PATH)
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bound_boxes(self, results: list, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

            for cls_id, conf in zip(boxes.cls, boxes.conf):
                self.class_confidences[int(cls_id)].append(float(conf))

        return results[0].plot(), xyxys, confidences, class_ids

    def calculate_summary(self):
        summary = {}
        for cls_id, confidences in self.class_confidences.items():
            average_conf = sum(confidences) / len(confidences) if confidences else 0
            class_name = CLASS_ID_TO_NAME.get(cls_id, "Unknown")  # Get class name
            summary[class_name] = {"count": len(confidences), "average_confidence": average_conf}
        return summary
    
    def filter_dominant_signs(self, summary):
        counts = [data["count"] for data in summary.values()]
        threshold = np.mean(counts)

        dominant_signs = {name: data for name, data in summary.items() if data["count"] >= threshold}
        return dominant_signs

    def __call__(self):
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), "Error: Could not open video source."

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            frame_resized = cv2.resize(frame, (640, 640))

            results = self.predict(frame_resized)

            annotated_frame, xyxys, confidences, class_ids = self.plot_bound_boxes(results, frame)

            cv2.imshow("Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        summary = self.calculate_summary()
        filename = os.path.splitext(os.path.basename(self.source))[0]

        filepath = f"Result/{filename}.txt"

        with open(filepath, "w") as file:
            file.write("Detected Signs:\n")
            for class_name, data in summary.items():
                file.write(f"Class Name: {class_name}, Count: {data['count']}, Average Confidence: {data['average_confidence']:.2f}\n")

            dominant_signs = self.filter_dominant_signs(summary)
            file.write("\nSummary:\n")
            for class_name, data in dominant_signs.items():
                file.write(f"Class Name: {class_name}, Count: {data['count']}, Average Confidence: {data['average_confidence']:.2f}\n")
    
        # for class_name, data in summary.items():
        #     print(f"Class Name: {class_name}, Count: {data['count']}, Average Confidence: {data['average_confidence']:.2f}")
    
if __name__ == "__main__":
    detector = ObjectDetection(r"documentation\ref\Video\vid (9).mp4")
    detector()