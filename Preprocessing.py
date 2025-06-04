"""
Creator: Mario Toscano and Francesco Stiro

Script to convert MOT17 dataset into yolo format and filter the class on gt.txt for people detection purpose

"""

import os
import shutil
import pandas as pd
from enum import Enum
from collections import defaultdict

"""
We want to filter the video directory. On our analysis we want to consider the pedastrian, the occupant and the stationary person
the only videos that have this information coded into the gt.txt file are the video in the "FRCNN" directory.
so we wanto to considerer only this directory

"""
DIR_SELECTOR = "FRCNN"


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

def valutation(objectToFilter):
    """
    This function allow us to obtain the directory where there are the frames of the video that contain all the class preferably to build the validation directory

    Args:
        objectToFilter (List<Object_GT_Code>): a list that contain the code to filter on gt.txt file

    Returns:
        the name of the directory that contain the video whit all the class preferably
    """
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
    class_per_video = defaultdict(set)

    sequences = [seq for seq in os.listdir(script_dir) if (os.path.isdir(os.path.join(script_dir, seq)) and DIR_SELECTOR in seq)]

    for seq in sequences:
        df = pd.read_csv(os.path.join(script_dir,seq,"gt","gt.txt"), header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']

        for cls in set(df["class"].unique()).intersection({object.value for object in objectToFilter}):
            class_per_video[seq].add(int(cls))

    for video, classes in class_per_video.items():
        print(f"{video}: {sorted(classes)}")

    val_video = max(class_per_video, key=lambda k: len(class_per_video[k]))
    if len(class_per_video[val_video])!=len(objectToFilter):
        print("ATTENTION, THERE ARE NOT ANY VIDEO THAT CONTAINS ALL THE LABELS. THE TRAINING COULD LEAD TO ERRORS!!")    
    print(f"\n{val_video} is choosen with {len(class_per_video[val_video])} classes\n")
    
    return val_video


def convert_mot_to_yolo_single_folder(dataset_root, output_root, objectToFilter, img_size=(1920, 1080), visibility_threshold=0.3):
    """
    This function converts the MOT7 dataset to a yolo format.

    Args:
        dataset_root: the path where the "train" directory is
        output_root: the path where we want to create the "output" directory
        objectToFilter: a list that contain the code to filter on gt.txt file
        img_size (tuple, optional): the size of the frame. Defaults to (1920, 1080).
        visibility_threshold (float, optional): the threshold of the visibility. we don't want to consider all the obejct with 0 visibility. Defaults to 0.3.

    """
    video_for_val = valutation(objectToFilter)
    images_output = os.path.join(output_root, 'images', 'train')
    val_images_output = os.path.join(output_root, 'images', 'val')
    labels_output = os.path.join(output_root, 'labels', 'train')
    val_labels_output = os.path.join(output_root, 'labels', 'val')
    switch = {1:0, 2:1, 7:2} #da rendere generale
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    os.makedirs(val_labels_output, exist_ok=True)
    os.makedirs(val_images_output, exist_ok=True)
    
    sequences = [seq for seq in os.listdir(dataset_root) if (os.path.isdir(os.path.join(dataset_root, seq)) and DIR_SELECTOR in seq) ]
    print(img_size[1])
    for seq in sequences:
        print(f"Processing sequence: {seq}")
        seq_path = os.path.join(dataset_root, seq)
        img_path = os.path.join(seq_path, 'img1')
        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        val_path = os.path.join(seq_path)

        if not os.path.isfile(gt_path) or not os.path.isdir(img_path):
            continue

        df = pd.read_csv(gt_path, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']

        df = df[(df['class'].isin([object.value for object in objectToFilter])) & (df['vis'] > visibility_threshold)]
         
        for _, row in df.iterrows():
            frame_id = int(row['frame'])
            image_name = f"{frame_id:06d}.jpg"
            image_src_path = os.path.join(img_path, image_name)
           
            new_image_name = f"{seq}_{frame_id:06d}.jpg"
            new_label_name = new_image_name.replace('.jpg', '.txt')
     
            new_image_path =  os.path.join(images_output, new_image_name) if seq!= video_for_val else os.path.join(val_images_output, new_image_name)
            if not os.path.exists(new_image_path):
                try:
                    shutil.copy(image_src_path, new_image_path)
                except FileNotFoundError:
                    print(f"Missing image: {image_src_path}")
                    continue
            if(row['x'] <0 or row['y']<0 or row['w']<0 or row['h']<0):
                continue  
            if(((row['x'] + row['w'] / 2) / img_size[0] )> 1 or ((row['y'] + row['h'] / 2) / img_size[1]) > 1):
                continue
            x_center = (row['x'] + row['w'] / 2) / img_size[0]
            y_center = (row['y'] + row['h'] / 2) / img_size[1]
            middlew= row["w"]
            middleh= row["h"]
            if(row["x"]+middlew>img_size[0]):
                middlew = img_size[0]-row["x"]
            if(row["y"]+middleh>img_size[1]):
                middleh = img_size[1]-row["y"]

            w = middlew / img_size[0]   
            h = middleh / img_size[1]
            class_label = switch.get(int(row["class"]))
            
            label_path = os.path.join(labels_output, new_label_name) if seq!= video_for_val else os.path.join(val_labels_output, new_label_name)
            
            with open(label_path, 'a') as f:
                f.write(f"{class_label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("Conversion completed!")

convert_mot_to_yolo_single_folder(
    dataset_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),'train'),           
    output_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),'yolo'),      
    img_size=(1920, 1080),                   
    visibility_threshold=0.0,               
    objectToFilter = [Object_GT_Code.Pedestrian, Object_GT_Code.Occupant, Object_GT_Code.StationaryPerson]
)
