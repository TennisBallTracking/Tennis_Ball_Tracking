import pandas as pd
import numpy as np
import cv2
from filterpy.common import kinematic_kf
from filterpy.kalman import IMMEstimator
from filterpy.stats import mahalanobis
import math
from copy import deepcopy
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

VIDEO_FILE = '../m-945.mp4'
FPS = 29.97
START_MS = 166000  # 260000 #234000 #192000 #166000 #148000 #123000 #88000
END_MS = 174500  # 268000 #240000 #204000 #174500 #152000 #132000 #98000

BALL_DETECTRON2_WEIGHTS = 'soccer-ball-20220607-2034.pth'
BALL_DETECTRON2_THRESHOLD = 0.98
DEBUG = True
MANUAL_CORRECTION = False
USE_CROP = True
MAX_CROP = 0.75

cap = cv2.VideoCapture(VIDEO_FILE)
cap.set(0, START_MS)
ret, frame = cap.read()
HEIGHT, WIDTH = frame.shape[0:2]
PER_FRAME = 1000/FPS

# Detectron2 Ball detector
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =BALL_DETECTRON2_THRESHOLD # Set threshold for this model
cfg.MODEL.WEIGHTS =  BALL_DETECTRON2_WEIGHTS
ball_detector = DefaultPredictor(cfg)

def colorHist(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def detector1(frame, ball_xy, ball_var):
    if USE_CROP:
        # Calc cropping area
        std_x = int(round(math.sqrt(ball_var[0])))
        std_y = int(round(math.sqrt(ball_var[1])))
        crop_width = max(std_x * 2, int(WIDTH * MAX_CROP))
        crop_width = min(crop_width, WIDTH)
        crop_height = max(std_y * 2, int(HEIGHT * MAX_CROP))
        crop_height = min(crop_height, HEIGHT)
        base_x, base_y = int(round(ball_xy[0] - crop_width / 2)), int(round(ball_xy[1] - crop_height / 2))
        base_x, base_y = max(0, base_x), max(0, base_y)
        base_x, base_y = min(WIDTH - crop_width, base_x), min(HEIGHT - crop_height, base_y)
        crop_frame = frame[base_y:base_y + crop_height, base_x:base_x + crop_width, :].copy()
    else:
        crop_frame = frame.copy()

    if DEBUG:
        cv2.imshow('crop_frame', crop_frame)

    output = ball_detector(crop_frame)
    ball_boxes = output['instances'][output['instances'].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
    ball_scores = output['instances'][output['instances'].pred_classes == 0].scores.cpu().numpy().reshape(-1,1)
    if USE_CROP and ball_boxes.size > 0:
        ball_boxes = ball_boxes + [base_x, base_y, base_x, base_y]
    ball_centers = np.concatenate(
        [(ball_boxes[:, 0:1] + ball_boxes[:, 2:3]) / 2, (ball_boxes[:, 1:2] + ball_boxes[:, 3:4]) / 2,
         ball_scores], axis=1)  # [[x,y,score]...]

    return ball_centers

fa_trackers = []  # TAs, Tracks for false Alarms
ta_template = kinematic_kf(dim=2, order=1, dt=1.0, order_by_dim=False)
ta_template.Q = np.diag([1.5, 1.5, 1.5, 1.5])
ta_template.R = np.diag([4.5, 4.5])
TA_GATING = 3.0
TA_COV_THRESHOLD = 100.0
TA_CLEAR_MARGIN = 1.2

# TBs, Tracks for the Ball
tb1 = kinematic_kf(dim=2, order=1, dt=1.0, order_by_dim=False)
tb1.Q = np.diag([2.5, 2.5, 2.5, 2.5])
tb1.R = np.diag([100.5, 100.5])   #14.5
tb1.x = np.array([[WIDTH/2, HEIGHT/2, 0., 0.]]).T
tb1.P = np.eye(4) * [(WIDTH/2) ** 2., (HEIGHT/2) ** 2, (WIDTH/2) ** 2., (HEIGHT/2) ** 2]

tb2 = kinematic_kf(dim=2, order=1, dt=1.0, order_by_dim=False)
tb2.F = np.array([[1., 0, 0, 0],
                  [0., 1, 0, 0],
                  [0., 0, 0, 0],
                  [0., 0, 0, 0]])
tb2.Q = np.diag([625, 625, 625, 625])
tb2.R = np.diag([100.5, 100.5])
tb2.x = np.array([[WIDTH/2, HEIGHT/2, 0., 0.]]).T
tb2.P = np.eye(4) * [(WIDTH/2) ** 2., (HEIGHT/2) ** 2, (WIDTH/2) ** 2., (HEIGHT/2) ** 2]

filters = [tb1, tb2]
M = np.array([[0.9, 0.1],
              [0.7, 0.3]])
mu = np.array([0.75, 0.25])
bank = IMMEstimator(filters, mu, M)

TB_GATING = np.array([6.0, 4.0]) #np.array([6.0, 3.0])
WEIGHT_SCORE = True

# TS, Track for Smoother
ts_F = np.average([t.F for t in filters], weights=mu, axis=0)
ts_Q = np.sum([t.Q for t in filters], axis=0)
ts_buffer_xs, ts_buffer_Ps = [], []

frame = np.ones(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)*255

def click_point(event, x, y, flags, param):
    global ts_buffer_xs, ts_buffer_Ps, bank, frame
    if event == cv2.EVENT_LBUTTONDOWN:  # to mark image points by left mouse click
        bank.update(np.array([[x, y]]).astype(np.float))
        cv2.circle(frame, (x, y), r, color, 2)
        cv2.imshow('frame', frame)
        del ts_buffer_xs[-1]
        del ts_buffer_Ps[-1]
        ts_buffer_xs.append(bank.x.copy())
        ts_buffer_Ps.append(bank.P.copy())

if MANUAL_CORRECTION:
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', click_point)

t = START_MS
cap.set(0, t)
prev_color_hist = None
while t < END_MS:
    # step 1
    bank.predict()

    ret, frame = cap.read()
    color_hist = colorHist(frame)
    if prev_color_hist is not None and cv2.compareHist(prev_color_hist, color_hist, cv2.HISTCMP_HELLINGER) > 0.5:
        # very different from prev frame, stop at this frame
        break
    prev_color_hist = color_hist

    ball_centers = detector1(frame, bank.x[0:2, 0], [bank.P[0, 0], bank.P[1, 1]])
    zs, cs = ball_centers[:, 0:2], ball_centers[:, 2]  # detections, scores

    if DEBUG:
        [cv2.circle(frame, tuple(c), 2, (0, 0, 255), 2) for c in np.round(zs).astype(int)]
        print(ball_centers)
        # Uncomment the following code to see the x,P for each TB
        x, y = int(round(filters[0].x[0, 0])), int(round(filters[0].x[1, 0]))
        r1, r2 = int(round(math.sqrt(filters[0].P[0, 0]))), int(round(math.sqrt(filters[0].P[1, 1])))
        cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (0, 255, 0), 1)
        x, y = int(round(filters[1].x[0, 0])), int(round(filters[1].x[1, 0]))
        r1, r2 = int(round(math.sqrt(filters[1].P[0, 0]))), int(round(math.sqrt(filters[1].P[1, 1])))
        cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (255, 0, 0), 1)

    # step 2, delete TA[i] if its cov is too large
    for i in range(len(fa_trackers))[::-1]:
        ta = fa_trackers[i]
        ta.predict()

        if DEBUG:
            # Uncomment the following code to see the x,P for each TA
            x, y = int(round(ta.x[0, 0])), int(round(ta.x[1, 0]))
            r1, r2 = int(round(math.sqrt(ta.P[0, 0]))), int(round(math.sqrt(ta.P[1, 1])))
            cv2.ellipse(frame, (x, y), (r1, r2), 0, 0, 360, (0, 0, 255), 1)

        if max(ta.P[0, 0], ta.P[1, 1]) > TA_COV_THRESHOLD:  # threshold set here
            del fa_trackers[i]
            if DEBUG:
                print('false alarm tracks: ', len(fa_trackers))

    # step 3, for each TA, assosiate with the nearest candidate, remove it
    if zs.size > 0:
        for i in range(len(fa_trackers)):
            ta = fa_trackers[i]
            if zs.size > 0:
                unassigned_dts = [i for i in range(zs.shape[0])]
                distances = [mahalanobis(zs[j, 0:2], ta.x[0:2], ta.P[0:2, 0:2]) for j in unassigned_dts]
                nearest = np.argmin(distances)
                if distances[nearest] < TA_GATING:
                    ta.update(zs[nearest, 0:2])
                    unassigned_dts.remove(nearest)
                    if DEBUG:
                        print(zs[nearest, 0:2], 'is false alarm, by ', ta.x[0, 0], ta.x[1, 0], ta.P[0, 0], ta.P[1, 1])
                    zs, cs = zs[unassigned_dts], cs[unassigned_dts]

    # step 4, B.update() using average ball in gate
    if zs.size > 0:
        un_associated, gated_balls, weights = [], [], []
        for i in range(zs.shape[0]):
            distances = np.array([mahalanobis(zs[i, 0:2], kf.x[0:2], kf.P[0:2, 0:2]) for kf in filters])
            if np.all((distances - TB_GATING) < 0):
                gated_balls.append(zs[i])
                if WEIGHT_SCORE:
                    weights.append(math.exp(-1 * np.min(distances)) * cs[i])
                else:
                    weights.append(math.exp(-1 * np.min(distances)))
                if DEBUG:
                    print(zs[i, 0:2], 'is a tracked ball')
            else:
                un_associated.append(i)
                if DEBUG:
                    print('distances:', distances)

        if len(gated_balls) > 0:
            gated_balls = np.array(gated_balls).reshape(-1, 2)
            avg_ball = np.average(gated_balls, axis=0, weights=weights)
            bank.update(avg_ball)
        zs, cs = zs[un_associated], cs[un_associated]

    # step 5, create additional TAs
    if zs.size > 0:
        for i in range(zs.shape[0]):
            ds = np.array([mahalanobis(zs[i, 0:2], kf.x[0:2], kf.P[0:2, 0:2]) for kf in filters])
            if np.all((ds - TB_GATING * TA_CLEAR_MARGIN) > 0):  # clearly outside all ball gating
                ta = deepcopy(ta_template)
                ta.x = np.array([[zs[i, 0], zs[i, 1], 0., 0.]]).T
                fa_trackers.append(ta)
                if DEBUG:
                    print('Add a false alarm tracks: ', len(fa_trackers))
                    print(zs[i, 0:2], 'is false alarm')

    # step 6
    ts_buffer_xs.append(bank.x.copy())
    ts_buffer_Ps.append(bank.P.copy())

    if DEBUG:
        # code to show the result
        x, y = int(round(bank.x[0, 0])), int(round(bank.x[1, 0]))
        if bank.mu[0] > mu[0]:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        r = max(2, int(round(math.sqrt(bank.P[0,0]))))
        cv2.circle(frame, (x, y), r, color, 2)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(int(PER_FRAME) * (MANUAL_CORRECTION != True))
        if k == ord('q'):
            break
    t += PER_FRAME

# step 6, do the smoothing at the end of the batch, it can be called on any tracker because ts_F, ts_Q is presented
smoothed_xs, smoothed_Ps, _, _ = tb1.rts_smoother(np.array(ts_buffer_xs).reshape(-1, 4, 1),
                                                  np.array(ts_buffer_Ps).reshape(-1, 4, 4),
                                                  np.array([ts_F] * len(ts_buffer_xs)).reshape(-1, 4, 4),
                                                  np.array([ts_Q] * len(ts_buffer_xs)).reshape(-1, 4, 4),)

t = START_MS
cap.set(0, t)
print('time_ms,x,y,var')
while t < min(END_MS, START_MS + int(len(ts_buffer_xs) * PER_FRAME) - 1):
    ret, frame = cap.read()
    est_ball = smoothed_xs[int(round((t-START_MS)/PER_FRAME))][0:2].reshape(2)
    est_var = int(round(math.sqrt(np.max(smoothed_Ps[int(round((t - START_MS) / PER_FRAME))][0:2, 0:2].reshape(2,2)))))
    print('%d,%0.2f,%0.2f,%0.2f'%(t, est_ball[0], est_ball[1], est_var))
    if DEBUG:
        cv2.circle(frame, (int(round(est_ball[0])), int(round(est_ball[1]))), 15, (0, 255, 255), 2)
        cv2.circle(frame, (int(round(est_ball[0])), int(round(est_ball[1]))), 2 * est_var, (0, 255, 255), 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(int(PER_FRAME))
    t += PER_FRAME
