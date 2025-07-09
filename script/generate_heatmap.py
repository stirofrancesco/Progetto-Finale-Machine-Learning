import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm


dataset_path = "../dataset"  
subfolders = ["train","val"]
canvas_size = (1280, 1280)  
heatmap = np.zeros(canvas_size, dtype=np.float32)
image_exts = [".jpg", ".jpeg", ".png"]


def add_bbox_to_heatmap(xc, yc, w, h, img_w, img_h):
    x = int(xc * canvas_size[1])
    y = int(yc * canvas_size[0])
    sigma = int(min(w * img_w, h * img_h)) or 2
    sigma = int(sigma * (canvas_size[0] / img_h)) 
    sigma = max(sigma, 2)
    patch = np.zeros(canvas_size, dtype=np.float32)
    cv2.circle(patch, (x, y), sigma, 1, -1)
    heatmap[:] += patch

for subset in subfolders:
    label_dir = os.path.join(dataset_path, "labels", subset)
    image_dir = os.path.join(dataset_path, "images", subset)
    label_files = glob(os.path.join(label_dir, "*.txt"))
    
    for label_path in tqdm(label_files, desc=f"Processing {subset}"):
        filename = os.path.basename(label_path).replace(".txt", "")

        image_path = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, filename + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        if not image_path:
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = map(float, parts)
                add_bbox_to_heatmap(xc, yc, w, h, img_w, img_h)

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3, sigmaY=3)
    heatmap_norm = heatmap / heatmap.max()
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap_norm, cmap="hot")
    plt.axis("off")
    plt.title("Heatmap aggregata delle annotazioni (normalizzata)")
    plt.tight_layout()
    plt.savefig(f"heatmap_annotazioni_{subset}.png", dpi=300)
    heatmap = np.zeros(canvas_size, dtype=np.float32)
