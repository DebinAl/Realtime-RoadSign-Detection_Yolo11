# Real-Time Road Sign Detection Using YOLOv11
This project demonstrates the implementation of YOLOv11 for real-time road sign detection, leveraging deep learning techniques to accurately identify and classify road signs in live video streams. The system is designed to assist in real-time navigation and improve road safety.

For a detailed explanation of the application's architecture, features, and technology stack, refer to the [Project Report](https://1drv.ms/w/c/7a8f2b2853bae142/EVi_EnlXqUVEjuUcC273Z_8BquUMs9jLFVDN27OXNq6k8w?e=FVxjUd).

## How to Run the Project
**1. Clone the Repository:**
```bash
git clone https://github.com/DebinAl/Realtime-RoadSign-Detection_Yolo11.git
cd Realtime-RoadSign-Detection_Yolo11
```

**2. Set Up Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Detection Application:**
- Open the app.py script.
- Modify the following line to set the input source (e.g., a video file or webcam):
```python
if __name__ == "__main__":
    detector = ObjectDetection(r"path/to/input")  # Change this to your desired input
    detector()
```
- Example Inputs:
  - Local video file: "path_to_video.mp4"
  - Webcam: 0 (use the integer 0 for default webcam)
- Save the changes and run the script:
```bash
python app.py

```
