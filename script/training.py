
import sys
import os

from ultralytics import YOLO

def main():
    model = YOLO('yolo12n.pt')
    results = model.train(
      data='../dataset/dataset.yaml',
      epochs=100, 
      batch=12, 
      imgsz=640,
      scale=0.9,  
      mosaic=0.4,
      close_mosaic = 10,
      #mixup=0.05,  
      #copy_paste=0.2, 
      device="0",
      workers = 4,
      amp = True,
      patience=20,
      verbose=True,
      cache = True,
      #cutmix=0.2,
      name="Dataset-yolov12s-640-crowd+Mot",
      #weight_decay= 0.001,
      resume = False
    )  


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  
    main()
