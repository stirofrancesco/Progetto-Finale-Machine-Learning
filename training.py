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
    model = YOLO('..\yolov12\\ultralytics\cfg\models\\v12\yolov12.yaml')

  # Train the model
    results = model.train(
      data='MOT17.yaml',
      epochs=30, 
      batch=8, 
      imgsz=640,
      scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
      mosaic=1.0,
      mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
      copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
      device="0",
    )

    #  Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("test.jpg")
    results[0].show()

import cv2
import os



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  
    main()
