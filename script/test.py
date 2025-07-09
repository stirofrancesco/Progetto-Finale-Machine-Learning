from ultralytics import YOLO

model = YOLO("yolo12n")  

metrics = model.val(data="../dataset/dataset.yaml", split="test",save=True, save_txt=True, save_conf=True, single_cls= True)
