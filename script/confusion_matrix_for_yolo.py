
from ultralytics import YOLO
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = "yolo12m"
output_image = "confusion_matrix_m_norm.png"

dataset = sv.DetectionDataset.from_yolo(
   images_directory_path="../dataset/images/test",
    annotations_directory_path= "../dataset/labels/test",
    data_yaml_path=f"../dataset/dataset.yaml"
)

model = YOLO(model)
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    mask = detections.class_id == 0
    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[mask],
        class_id=detections.class_id[mask],
        confidence=detections.confidence[mask]
    )
    return filtered_detections

confusion_matrix = sv.ConfusionMatrix.benchmark(
   dataset = dataset,
   callback = callback
)

custom_class_labels = ["Person", "background"]
matrix = confusion_matrix.matrix.astype(np.int32)
matrix_rotated = matrix.T
row_sums = matrix_rotated.sum(axis=0, keepdims=True)
normalized_matrix = matrix_rotated / (row_sums + 1e-6)

plt.figure(figsize=(8, 6))
sns.heatmap(
    normalized_matrix,
    annot=True,
    fmt=".2f",  
    cmap="Blues",
    xticklabels=custom_class_labels,
    yticklabels=custom_class_labels
)
plt.xlabel("True")
plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.show()



