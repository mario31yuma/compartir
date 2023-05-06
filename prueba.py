from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
#model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.pt')
# model = YOLO('final.pt')
model = YOLO('30_04entreno1.pt')

# Use the model
#results = model('Videos/video1.mov', save=True)  # predict on an image

#results = model.track(source="Videos/video3.mp4", show=True, save=True, conf=0.20) # ¿device=cpu?
results = model.track(source="Videos/video1.mov", show=True, save=True, conf=0.20)

#print(results)
#print(results[0][3])