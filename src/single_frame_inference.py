# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import pandas as pd
import cv2
import os
import math
from tqdm import tqdm as tqdm

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Calculate distance
def get_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


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
tp, fp, fn = 0, 0, 0
for index, row in tqdm(df_test.iterrows()):
    time_ms = int(row.time_ms)
    match_id = int(row.match_id)
    x_gt, y_gt = row.x, row.y
    frame = cv2.imread('../train_imgs/%d_%d.jpg'%(match_id, time_ms))
    outputs = predictor(frame)
    pred_box = outputs['instances'].to('cpu').pred_boxes.tensor.detach().numpy()
    scores = outputs['instances'].to('cpu').scores.detach().numpy()
    score = 0
    if len(pred_box) == 0:
        result.append([match_id, time_ms, -1, -1, 0])
        fn += 1
        continue
    else:
        final_x, final_y, max_score = -1, -1, 0
        for i, final_pred in enumerate(pred_box):
            x, y = int((final_pred[0] + final_pred[2]) / 2), int((final_pred[1] + final_pred[3]) / 2)
            score = scores[i]
            if score > max_score:
                max_score = score
                final_x, final_y = x, y
            result.append([match_id, time_ms, row.ball_sn, x, y, score])
        if get_dist([final_x, final_y], [x_gt, y_gt]) <= 20:
            tp += 1
        else:
            fp += 1

result = pd.DataFrame(result, columns=['match_id', 'time_ms', 'ball_sn', 'x', 'y', 'score'])
result.to_csv('m-%03d-single.csv', index=False)

print('TP: %d, FP: %d, FN: %d'%(tp, fp, fn))
print('Accuracy: %f'%(tp / (tp + fp + fn)))
