import os
import cv2
import numpy as np
from PIL import Image

def clamp(x,y,w,h,img_width=1920,img_height=1080):
    if x < 0:
        w += x  
        x = 0

    if y < 0:
        h += y
        y = 0

    if x + w > img_width:
        w = img_width - x

    if y + h > img_height:
        h = img_height - y

    return x,y,w,h   


'''
    Create dataset structure in this format for yolo
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
'''
def create_dataset_structure(dataset_dir):
    os.makedirs(dataset_dir, exist_ok = True)
    
    train_dir_images = os.path.join(dataset_dir,"images", "train")
    train_dir_labels = os.path.join(dataset_dir,"labels", "train")
    val_dir_images = os.path.join(dataset_dir,"images", "val")
    val_dir_labels = os.path.join(dataset_dir,"labels", "val")
    
    os.makedirs(train_dir_images, exist_ok=True)
    os.makedirs(train_dir_labels, exist_ok=True)
    os.makedirs(val_dir_images, exist_ok=True)
    os.makedirs(val_dir_labels, exist_ok=True)
    return train_dir_images, train_dir_labels, val_dir_images, val_dir_labels

''' Used to print labels on Image '''

def create_labels_on_image(img, txt):
    
    h, w, _ = img.shape

    box_w = int(w * 0.3)      
    box_h = int(h * 0.08)     

    center_x = w // 2

    start_point = (center_x - box_w // 2, 10)
    end_point = (center_x + box_w // 2, 10 + box_h)
    

    cv2.rectangle(img, start_point, end_point, color=(0, 0, 0), thickness=-1)  

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    while True:
        (text_w, text_h), _ = cv2.getTextSize(txt, font_face, font_scale, thickness)
        if text_w < box_w - 20 and text_h < box_h - 10:
            font_scale += 0.1
        else:
            font_scale -= 0.1
            break

    (text_w, text_h), _ = cv2.getTextSize(txt, font_face, font_scale, thickness)

    center_x = start_point[0] + box_w // 2
    center_y = start_point[1] + box_h // 2

    text_x = int(center_x - text_w / 2)
    text_y = int(center_y + text_h / 2 - 5)

    cv2.putText(img, txt, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness)

    return img

''' Used to print the bbox on Image '''

def draw_gt(frame, gt, class_colors=None):

    img = np.array(frame)
    h, w = img.shape[:2]
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    height, width = img.shape[:2]

    if class_colors is None:
        class_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}  


    for line in gt:
        parts = line.strip().split()
        if len(parts) != 5:
            
            continue  

        class_id, x_center, y_center, w, h = parts
        class_id = int(class_id)
        x_center, y_center, w, h = map(float, (x_center, y_center, w, h))

        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = class_colors.get(class_id, (255, 255, 255))  

        cv2.rectangle(img, (x1, y1), (x2, y2), color,max(1, int(round(min(w, h) * 0.0025))) )

        text = "persone" if (len(gt))>1 else "persona"
    return create_labels_on_image(img,f"{len(gt)} {text}")



