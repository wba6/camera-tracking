# yolo.py

from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n_ncnn_model", task='detect')

