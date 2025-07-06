"""
Creator: Mario Toscano and Francesco Stiro

Script to convert MOT dataset into yolo format and filter the class on gt.txt for people detection purpose

"""

import os
import shutil
import pandas as pd
from enum import Enum
from collections import defaultdict
from PIL import Image
from utils import clamp, create_dataset_structure
import cv2
from tqdm import tqdm

MOT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Mot_raw')
OUTPUT_YOLO_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dataset')

class Object_GT_Code(Enum):
    """
    This enum class rappresent the different class of MOT17 that allow to represent uman on different situation.
    The different value are coded for MOT17 class according to gt.txt structure:

    1	Pedestrian
    2	Person on vehicle
    3	Car
    4	Bicycle
    5	Motorcycle
    6	Non-motorized vehicle
    7	Static person
    8	Distractor
    9	Occluder
    10	Occluder on the ground
    11	Occluder full
    12	Reflection

    """
    Pedestrian = 1
    Occupant = 2
    StationaryPerson = 7    

def convert_mot_to_yolo_single_folder(dataset_root, output_root, objectToFilter, visibility_threshold=0.3, frame_to_skip = 12 ):
    """
        This function converts the MOT17 and MOT20 dataset to a yolo format.

    Args:
        dataset_root: the path where the "train" directory is
        output_root: the path where we want to create the "output" directory
        objectToFilter: a list that contain the code to filter on gt.txt file
        visibility_threshold (float, optional): the threshold of the visibility. we don't want to consider all the obejct with 0 visibility. Defaults to 0.3.
    """
   
    images_output,labels_output,val_images_output,val_labels_output = create_dataset_structure(output_root)
   
    sequences = [seq for seq in os.listdir(dataset_root) if (os.path.isdir(os.path.join(dataset_root, seq)) ) ]
    video_for_val = ["MOT17-04-FRCNN", "MOT17-02-FRCNN"]
    original_w=1920
    original_h=1080
    for seq in sequences:
    
        seq_path = os.path.join(dataset_root, seq)
        img_path = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        val_path = os.path.join(seq_path)
        
        if not os.path.isfile(gt_path) or not os.path.isdir(img_path):
            continue

        df = pd.read_csv(gt_path, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
        ''' we filters the class and and his visibility '''
        df = df[(df['class'].isin([object.value for object in objectToFilter])) & (df['vis'] > visibility_threshold)]
        ''' there we skip the frame for reduce dataset and skip similarity '''
        df = df[df['frame'] % frame_to_skip == 0].reset_index(drop=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {seq}"):
            frame_id = int(row['frame'])
            image_name = f"{frame_id:06d}.jpg"
            image_src_path = os.path.join(img_path, image_name)
           
            new_image_name = f"{seq}_{frame_id:06d}.jpg"
            new_label_name = new_image_name.replace('.jpg', '.txt')
     
            new_image_path =  os.path.join(images_output, new_image_name) if seq not in video_for_val else os.path.join(val_images_output, new_image_name)
            
            if not os.path.exists(new_image_path):
                img = cv2.imread(image_src_path)
                if img is None:
                    print(f"Failed to load image: {image_src_path}")
                    continue
                original_h,original_w  = img.shape[:2] 
                
                cv2.imwrite(new_image_path, img)

            x_new, y_new, middlew, middleh = clamp(row['x'],row['y'],row['w'],row['h'],img_width=original_w, img_height=original_h)
            if middlew <= 0 or middleh <= 0:
                print(f"Skipped box with non-positive size: {middlew}, {middleh}")
                continue            
                           
            x_center = (x_new + middlew / 2) / original_w
            y_center = (y_new + middleh / 2) / original_h          
            if(x_new+middlew>original_w):
                middlew = original_w-x_new
            if(y_new+middleh>original_h):
                middleh = original_h-y_new

            w = middlew / original_w   
            h = middleh / original_h
            class_label = 0
            
            label_path = os.path.join(labels_output, new_label_name) if seq not in video_for_val else os.path.join(val_labels_output, new_label_name)
            
            with open(label_path, 'a') as f:
                f.write(f"{class_label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("Conversion completed!")

convert_mot_to_yolo_single_folder(
    dataset_root=MOT_DATASET_ROOT,           
    output_root=OUTPUT_YOLO_DATASET,                   
    visibility_threshold=0.0,               
    objectToFilter = [Object_GT_Code.Pedestrian, Object_GT_Code.Occupant, Object_GT_Code.StationaryPerson]
)
