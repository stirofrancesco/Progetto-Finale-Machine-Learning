from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.predict import get_sliced_prediction
import cv2
from PIL import Image
import numpy as np
from tkinter import ttk
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from .utils import create_labels_on_image


CONF = 0.35

def sahi_prediction(model_path, frame, slice_height = 640, slice_width = 640):
    h, w = frame.shape[:2]
    annotator = Annotator(frame.copy(), line_width=max(1, int(round(min(w, h) * 0.0025))))
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=CONF,
        device='cpu'
    )
    results = get_sliced_prediction(frame, detection_model,overlap_height_ratio=0.2,overlap_width_ratio=0.2, slice_height =slice_height,slice_width = slice_width ,perform_standard_pred=False)
    detection_data = [
                (det.category.id,(det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
                for det in results.object_prediction_list
            ]
    for det in detection_data:
        annotator.box_label(det[1], label="", color =(255, 0, 0))

    text = "persone" if (len(detection_data))>1 else "persona"
    img = create_labels_on_image(annotator.result(),f"{len(detection_data)} {text}")
    img_PIL = Image.fromarray(img)
    img = Image.fromarray(img)
    img.save("prova.jpg")
    return img

def yolo_prediction(model_path, frame):
    model = YOLO(model_path)
    results = model.predict(source=frame,show= False , save = False , conf = CONF ,show_labels = False, show_conf = False, line_width = 1)
    img = results[0].orig_img.copy()  
    boxes = results[0].boxes  
    h, w = frame.shape[:2]
    for box in boxes:
        xyxy = box.xyxy[0].int().tolist() 
        color = (0, 255, 0)
        thickness = max(1, int(round(min(w, h) * 0.0025)))
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)

    text = "persone" if (len(boxes))>1 else "persona"
    img = create_labels_on_image(img,f"{len(boxes)} {text}")
    img_PIL = Image.fromarray(img)
    return img_PIL
