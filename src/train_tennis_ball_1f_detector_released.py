# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pandas as pd

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

OUTPUT_FOLDER = '../models/single_image_models'

def get_ball_dicts():
    df = pd.read_csv('../csv_files/train_label.csv')
    dataset_dicts = []
    for idx, row in tqdm(df.iterrows()):
        record = {}
        match_id = int(row.match_id)
        time_ms = int(row.time_ms)
        filename = '../train_imgs/%d_%d.jpg'%(match_id, time_ms)
        height, width = cv2.imread(filename).shape[:2]
        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        ## TODO: you can change the bounding box size accordingly
        x, y = int(row.x) - 10, int(row.y) - 10
        w, h = 20, 20
        ##
        objs = []
        obj = {
            'bbox': [x, y, w, h],
            'bbox_mode': BoxMode.XYWH_ABS,
            'category_id': 0,
        }
        objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Customized dataset
for d in ['train']:
    DatasetCatalog.register('ball_' + d, lambda d=d: get_ball_dicts())
    MetadataCatalog.get('ball_' + d).set(thing_classes=['ball'])
ball_metadata = MetadataCatalog.get('ball_train')

# Train
cfg = get_cfg()
cfg.merge_from_file('./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
cfg.DATASETS.TRAIN = ('ball_train',)
cfg.DATASETS.TEST = ()
# the number of layers you want to freeze in Resnet
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 60000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.5, 0.5]
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.OUTPUT_DIR = OUTPUT_FOLDER # can specify the output directory name
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
# if you want to resume training, change to "resume=True"
trainer.resume_or_load(resume=False)
trainer.train()

