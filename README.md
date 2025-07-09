# YOLOv12 - People Detection and Counting

## 📌 Overview

This project aims to detect and count people in video frames using the YOLOv12 object detection model. The main goal is to train and evaluate the performance of different YOLOv12 variants (`yolov12-n`, `yolov12-s`, and `yolov12-m`) on crowded scenes to achieve accurate and real-time pedestrian detection.

---

## 📁 Project Structure

```
├── script/                                 #contains the python script
│   ├── _pycache_/     
│   ├── gui_utils/              
│   │   ├── cache/
│   │   └── weigths/
│   │      ├──our_yolo12m.pt
│   │      ├──our_yolo12n.pt
│   │      └──our_yolo12s.pt
│   ├── CUHk_to_yolo.py
│   ├──__init__.py
│   ├──confusion_matrix_for_yolo.py
│   ├──crowd-human-to-yolo.py
│   ├──mot_to_yolo.py
│   ├──predict_for_gui.py
│   ├──test.py
│   ├──training.py
│   ├──utils.py
│   └──yolo11n.pt
├───test_result                       #contains the valutation on test set
│   ├───our_yolo_m
│   ├───our_yolo_n
│   ├───our_yolo_s
│   ├───yolo_m
│   ├───yolo_n
│   └───yolo_s
├───training_result                  #contains the result on training 
│   ├───Yolo12m
│   │   └───weights
│   ├───Yolo12n
│   │   └───weights
│   └───Yolo12s
│       └───weights
└───video_test                      #contains video for testing 
├───demo.mp4                        #short video that explains how to use the app
├───demo.py                         #app that allows you to do inference
├───download_dataset.py             #script that allows you to download the dataset
└───requirements.txt                #contains the dependencies
```

---

## 🧠 Model: YOLOv12

YOLOv12 is a state-of-the-art object detection model that enhances previous versions by integrating attention mechanisms, achieving improved accuracy without compromising inference speed. In this project, we focus on detecting a single class: Person.
Additionally, we employ SAHI (Sliced Aware Hyper Inference) to optimize detection performance on very small objects by performing inference on sliced image windows.

---

## 📊 Dataset

We used a combination of open-source pedestrian datasets:

- **MOT17**
- **MOT20**
- **CrowdHuman**
- **CUHK-Pedestrian**

Each dataset was preprocessed to:

- Convert annotations to YOLO format
- Normalize bounding boxes
- Merge multiple person-related classes into a single `Person` class
- Split into training and validation sets

---

## ⚙️ Training

To train a model:

```bash
cd script
python training.py 
```

Variants used:

- `yolov12-n` (nano): fast, lightweight
- `yolov12-s` (small): balanced
- `yolov12-m` (medium): more accurate, heavier

---

## ✅ Evaluation

Each model was evaluated using:

- **Precision-Recall Curve**
- **F1 vs Confidence**
- **Recall vs Confidence**
- **Precision vs Confidence**
- **Confusion Matrix (normalized and raw)**

---

## 🔍 Inference

You can use our demo to run the inference
```bash
python demo.py
```
The demo allows real-time video processing, analyzing the input frame by frame. You can pause the video at any frame if you don't need to process the entire sequence.

Additionally, the demo provides the option to save the output — either as a video or individual images — to a designated folder.

The video below demonstrates how to use the demo.
You can also integrate SAHI to perform inference on sliced windows, which is particularly useful for detecting very small objects.


## 🔴 Watch the Demo on YouTube

[![Demo Video](https://img.youtube.com/vi/jHO9VmmBf5k/0.jpg)](https://youtu.be/jHO9VmmBf5k)

Output will include:
- Bounding boxes
- Person count per frame
- Annotated images/videos

---

## 📈 Results

- **Best model**: YOLOv12-m
- **Application**: Accurate people counting in crowded public scenes
- **Weakness**: Confusion with the background in the presence of occlusions and occluded people
- **Future work**: Integration with tracking (e.g., DeepSORT) to handle video sequences

---

## 🧪 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:
```bash
python download_dataset.py
```

---

## 📬 Contact

For any questions, contact:

**Francesco Stiro**  
M.Sc.student - Computer Science

**Mario Toscano**  
M.Sc.student - Computer Science

