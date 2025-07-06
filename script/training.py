
import sys
print(sys.executable)
import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



from ultralytics import YOLO

def video_to_frames(video_path, prefix='frame'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore nell'apertura del video: {video_path}")
        return

    frame_count = 0
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fine del video
        frames.append(frame)      
        
    success, encoded_image = cv2.imencode('.jpg', frames[0])
    if success:
      jpeg_bytes = encoded_image.tobytes()
    return jpeg_bytes

def main():
    model = YOLO('yolo12n.pt')

    results = model.train(
      data='../dataset/dataset.yaml',
      epochs=100, 
      batch=12, 
      imgsz=640,
      scale=0.9,  # S:0.9; M:0.9; L:0.9; X:0.9
      mosaic=0.4,
      close_mosaic = 10,
      #mixup=0.05,  # S:0.05; M:0.15; L:0.15; X:0.2
      #copy_paste=0.2,  # S:0.15; M:0.4; L:0.5; X:0.6
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
    

    #Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    #results = model("running.mp4",save=True, show=True)
    

import cv2




if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  
    main()
