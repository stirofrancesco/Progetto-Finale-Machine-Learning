"""
Creator: Mario Toscano and Francesco Stiro

Script to convert Crowd Humans dataset into yolo format for people detection purpose

"""

import os
import json
from PIL import Image
import shutil
from tqdm import tqdm
from utils import create_dataset_structure, clamp

'''
percorso per i file da utilizzare

input_dataset_dir è la cartella in cui estraiamo il dataset
output_dataset_dir è la cartella in cui vogliamo creare il nostro dataset ricordandoci della struttura usata da yolo
''' 

input_dataset_dir = "../imagesCrowd/"                 
output_dataset_dir = "../dataset"

#creiamo le cartelle se non esistono
train_dir_images, train_dir_labels, val_dir_images, val_dir_labels = create_dataset_structure(output_dataset_dir)

#abbiamo una sola classe quidni mettiamo tutto a classe 0
class_id = 0  

train_annotations = []
val_annotations = []

for file in os.listdir(input_dataset_dir):
    if file.endswith(".odgt"):        
        if "val" in file:
            full_path = os.path.join(input_dataset_dir, file)
            with open(full_path, 'r') as f:
                val_annotations.extend(f.readlines())   
        else:
            full_path = os.path.join(input_dataset_dir, file)
            with open(full_path, 'r') as f:
                train_annotations.extend(f.readlines())
            

annotations = [train_annotations, val_annotations] 
for ann in annotations:

    desc = "Elaborazione validation"
    label_dir = val_dir_labels
    image_dir = val_dir_images

    if ann == train_annotations:
        desc = "Elaborazione training"
        label_dir = train_dir_labels
        image_dir = train_dir_images

    for line in tqdm(ann, desc):
        entry = json.loads(line.strip())

        img_name = entry["ID"] + ".jpg"
        img_path = os.path.join(input_dataset_dir,"Images",img_name)

        if not os.path.isfile(img_path):
            continue  

        img = Image.open(img_path)
        img_w, img_h = img.size

        label_file = os.path.splitext(img_name)[0] + ".txt"
        
        label_path = os.path.join(label_dir, label_file)
        shutil.copy(img_path, image_dir)

        with open(label_path, 'w') as out_f:
            for box in entry["gtboxes"]:
                if box.get("extra", {}).get("ignore", 0) == 1:
                    continue
                #usiamo il tag 'fbox' per riferirci alle bbox che rappresentano l'intero corpo
                x, y, w, h = box["fbox"]
                #sistemiamo le box che escono fuori dell'immagine
                x,y,w,h = clamp(x,y,w,h, img_w, img_h)
                #normalizziamo per il formato accettato da yolo
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                #scriviamo nel file nel formato yolo class x_center y_center width height
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
