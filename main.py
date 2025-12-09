from ultralytics import YOLO
import pandas as pd


source="squat2.mov"  # Sample image for testing
model = YOLO("yolo11n-pose.pt")
class_names = model.names  # Store model classes names
results = model(source=source, stream=True)

data = []

for result in results:
    kpt_list = result.keypoints.xy                 # keypoints for all objects
    for kpt in kpt_list:  
        for x, y in kpt.tolist(): 
            data.append({
                'x': x,
                'y': y
            })
        
df = pd.DataFrame(data)
print(df.head(20))