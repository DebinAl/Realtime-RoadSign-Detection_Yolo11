from roboflow import Roboflow

rf = Roboflow(api_key="KH2zEByuDjhHJhFGZ5y1")
project = rf.workspace("putri-mawaring-wening-lwwcx").project("traffic-sign-in-indonesia-detection")
version = project.version(3)
dataset = version.download("yolov11")