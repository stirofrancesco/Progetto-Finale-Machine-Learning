import os
import cv2
import scipy
import scipy.io
import numpy as np
from utils import create_dataset_structure

INPUT_CUHK_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','CHUK_dataset')    
OUTPUT_YOLO_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','dataset2')

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def parse_mat_vbb(vbb_path, img_width, img_height):
    mat = scipy.io.loadmat(vbb_path)
    vbb = mat['A'][0, 0]

    objLists = vbb['objLists']  
    n_frames = objLists.shape[1]
    annotations = {}

    for frame_idx in range(n_frames):
        objs = objLists[0, frame_idx]
        if objs.size == 0:
            continue

        for obj in objs[0]:
            pos = obj['pos'][0]  

            if pos.shape != (4,) or np.all(pos == 0):
                continue

            xtl, ytl, w, h = [float(v) for v in pos]

            xbr = xtl + w
            ybr = ytl + h

            xc = ((xtl + xbr) / 2) / img_width
            yc = ((ytl + ybr) / 2) / img_height
            ww = w / img_width
            hh = h / img_height

            cls = 0  
            annotations.setdefault(frame_idx, []).append((cls, xc, yc, ww, hh))

    return annotations


from tqdm import tqdm

def extract_frames_and_bbox(video_path, vbb_path, output_img_dir, output_txt_dir, v_name):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Impossibile aprire {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotations = parse_mat_vbb(vbb_path, width, height)

    for frame_idx in tqdm(range(total_frames), desc=f"Estrai {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        img_save = os.path.join(output_img_dir, f"{v_name}_{frame_idx:04d}.png")
        cv2.imwrite(img_save, frame)

        txt_save = os.path.join(output_txt_dir, f"{v_name}_{frame_idx:04d}.txt")
        with open(txt_save, 'w') as f:
            for cls, xc, yc, w, h in annotations.get(frame_idx, []):
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    cap.release()
    print(f"Processati {frame_idx+1} frame da {video_path}")

if __name__ == "__main__":
    
    video_folder = os.path.join(INPUT_CUHK_DATASET,"videos")
    vbb_folder = os.path.join(INPUT_CUHK_DATASET,"labels")

    train_image_folder, train_labels_folder, val_image_folder, val_labels_folder = create_dataset_structure(OUTPUT_YOLO_DATASET)

    video_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(".mp4")
    ]
    video_files.sort(key=lambda f: get_frame_count(os.path.join(video_folder, f)), reverse=True)
    i = 0
    for fname in video_files:
        if not fname.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(video_folder, fname)
        base = os.path.splitext(fname)[0]
        vbb_path = os.path.join(vbb_folder, base + ".vbb")
        if not os.path.exists(vbb_path):
            print(f"Attenzione: manca {vbb_path}")
            continue
        output_img_dir = val_image_folder if i==0 or i==7 else train_image_folder
        output_folder_label = val_labels_folder if i==0 or i==7  else train_labels_folder
        extract_frames_and_bbox(
            video_path,
            vbb_path,
            output_img_dir,
            output_folder_label,
            base
        )
        i += 1
