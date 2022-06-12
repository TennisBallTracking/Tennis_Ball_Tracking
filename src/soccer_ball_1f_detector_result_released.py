import numpy as np
import pandas as pd
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

visibilities = [1,2,3]   # 1=flying ball on complex background to make it nearly invisible, 2=occuluded, 3=visible, 0=out of view
DT_THRESHOLD = 0.98
DT_WEIGHTS = 'soccer-ball-20220607-2034.pth'
images_folder = './images5/'
DEBUG = False

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DT_THRESHOLD
cfg.MODEL.WEIGHTS = DT_WEIGHTS
cfg.MODEL.DEVICE = 'cuda'  # cuda or cpu
ball_detector1 = DefaultPredictor(cfg)

test_clips = [[904, 3915000, 3923866, 30], [904, 4620000, 4627766, 30], [904, 4919000, 4927866, 30], [904, 5902040, 5909773, 30], [904, 7620000, 7626066, 30],
              [908, 61680, 67040, 25], [908, 75000, 82400, 25], [908, 91680, 106880, 25], [908, 108360, 118440, 25], [908, 120760, 136040, 25],
              [945, 7000, 15000, 29.97], [945, 88000, 97400, 29.97], [945, 123000, 130000, 29.97], [945, 166000, 174500, 29.97], [945, 56000, 67500, 29.97]]
match_ids = {904,908,945}
distance_threshold = {904:15, 908:15, 945:20}

test_clips = np.array(test_clips)
for match_id in match_ids:
    correct, fp, fn = 0, 0, 0
    for MATCH_ID, START_MS, END_MS, FPS in test_clips[test_clips[:,0]==match_id]:
        ball_gt = pd.read_csv('m-%03d-ball-gt-%d-%d.csv' % (MATCH_ID, START_MS, END_MS))
        # cap = cv2.VideoCapture('../m-%03d.mp4' % MATCH_ID)
        for index, row in ball_gt[ball_gt.visible.isin(visibilities)].iterrows():
            # cap.set(0, row.time_ms)
            # ret, frame = cap.read()
            frame = cv2.imread(images_folder+'m-%03d-%08d.jpg'%(MATCH_ID, row.time_ms))

            output = ball_detector1(frame)
            ball_boxes = output['instances'][output['instances'].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
            if ball_boxes.size > 0:
                if DEBUG:
                    rounded_boxes = np.round(ball_boxes).astype(int)
                    [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (0, 0, 255), 1) for box in rounded_boxes]
                ball_centers = np.concatenate(
                    [(ball_boxes[:, 0:1] + ball_boxes[:, 2:3]) / 2, (ball_boxes[:, 1:2] + ball_boxes[:, 3:4]) / 2], axis=1)
                distances = np.linalg.norm(ball_centers - [row.x, row.y], axis=1)
                if np.min(distances) < distance_threshold[MATCH_ID]:
                    correct += 1
                fp += (np.sum(distances >= distance_threshold[MATCH_ID]))
            else:
                fn += 1

            if DEBUG:
                cv2.imshow('frame', frame)
                cv2.waitKey(int(1000/FPS))
    print('%d,%d,%d,%d,acc=%0.4f'%(match_id,correct,fp,fn,correct/(correct+fp+fn)))