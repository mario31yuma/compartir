import numpy as np
import os
from tqdm import tqdm

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import fiftyone.utils.random as four

from ultralytics import YOLO


#############################################################################
def export_yolo_data(
    samples, 
    export_dir, 
    classes, 
    label_field = "ground_truth", 
    split = None
    ):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples, 
                export_dir, 
                classes, 
                label_field, 
                split
            )   
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )
#############################################################################
# Cargamos un modelo de detección y de segmentación
# detection_model = YOLO("yolov8n.pt")
# seg_model = YOLO("yolov8n-seg.pt")

# Podemos hacer una inferencia de la siguiente manera
# detection_model("PATH_IMG")
# Sobre esto se nos genera un objeto result del tipo:
# ultralytics.yolo.engine.results.Results

## set classes to just include birds
classesBall = ["sports ball"]
classesPerson = ["person"]
classesComplete = ["sports ball", "person"]

train_dataset = foz.load_zoo_dataset(
    'coco-2017',
    split='train',
    classes=classesPerson,
    max_samples=20000
).clone()

train_dataset.name = "03_05entreno1"
train_dataset.persistent = True
train_dataset.save()

oi_samples = foz.load_zoo_dataset(
    "open-images-v6",
    classes = ["Ball"], # Añadir "Person" para cargar la clase de OI
    only_matching=True,
    label_types="detections",
    max_samples=10000
).map_labels(
    "ground_truth",
    {"Ball":"sports ball"} # Añadir "Person":"person" para renombrar la clase OI con la de Coco
)

oi_samples_2 = foz.load_zoo_dataset(
    "open-images-v6",
    classes = ["Person"],
    only_matching=True,
    label_types="detections",
    max_samples=15000
).map_labels(
    "ground_truth",
    {"Person":"person"} # Añadir "Person":"person" para renombrar la clase OI con la de Coco
)

train_dataset.merge_samples(oi_samples)
train_dataset.merge_samples(oi_samples_2)

## delete existing tags to start fresh
train_dataset.untag_samples(train_dataset.distinct("tags"))

## split into train and val
four.random_split(
    train_dataset,
    {"train": 0.8, "val": 0.2}
)

## export in YOLO format
export_yolo_data(
    train_dataset, 
    "03_05entreno1", 
    classesComplete, 
    split = ["train", "val"]
)



