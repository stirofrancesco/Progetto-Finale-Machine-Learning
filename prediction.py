from ultralytics import YOLO

model = YOLO("weight\\best.pt")

results = model.predict(source='prova6.jpg', save=True)
