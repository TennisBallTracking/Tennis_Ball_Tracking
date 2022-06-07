# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import pandas as pd
import cv2
import os
from tqdm import tqdm as tqdm

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

MATCH_ID = 20
CSV_FILE = '../csv_files/m-%03d-ball-gt.csv'%(MATCH_ID)
MODEL_FILE = '../models/single_image_models/model_final.pth'
cfg = get_cfg()
cfg.merge_from_file('./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# specify the model
cfg.MODEL.WEIGHTS = os.path.join(MODEL_FILE)
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.DATASETS.TEST = ('ball_val', )
predictor = DefaultPredictor(cfg)

df_test = pd.read_csv(CSV_FILE)
result = []
for index, row in tqdm(df_test.iterrows()):
    time_ms = int(row.time_ms)
    match_id = int(row.match_id)
    frame = cv2.imread('../train_imgs/%d_%d.jpg'%(match_id, time_ms))
    outputs = predictor(frame)
    pred_box = outputs['instances'].to('cpu').pred_boxes.tensor.detach().numpy()
    scores = outputs['instances'].to('cpu').scores.detach().numpy()
    score = 0
    if len(pred_box) == 0:
        result.append([match_id, time_ms, -1, -1, 0])
        continue
    else:
        for i, final_pred in enumerate(pred_box):
            x, y = int((final_pred[0] + final_pred[2]) / 2), int((final_pred[1] + final_pred[3]) / 2)
            score = scores[i]
            result.append([match_id, time_ms, row.ball_sn, x, y, score])

result = pd.DataFrame(result, columns=['match_id', 'time_ms', 'ball_sn', 'x', 'y', 'score'])
result.to_csv('m-%03d-single.csv', index=False)
