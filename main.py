from ultralytics import YOLO
import cv2


source="squat2.mov"  # Sample image for testing

# ==================================================
# Pose estimation: includes, class index, bounding
# box in xyxy, xywh format, keypoints and confidence
# ==================================================
model = YOLO("yolo11n-pose.pt")
class_names = model.names  # Store model classes names
results = model.predict(source=source, show=True, save=True)

for result in results:
    bbox_list = result.boxes.xyxy.tolist()          # bounding boxes all objects, you can also get xywh with boxes.xywh
    clss_list = result.boxes.cls.int().tolist()     # class index all objects
    conf_list = result.boxes.conf.tolist()          # confidence list all objects
    kpt_list = result.keypoints.xy                 # keypoints for all objects
    for box, kpt, cls, conf in zip(bbox_list, kpt_list, clss_list, conf_list):  # bbox, kpt, cls, id & conf
        print(f"Bounding box: {box}, Class index: {cls}, Class name: {class_names[cls]}, "
              f"Keypoints data: {kpt}, Confidence: {conf}")