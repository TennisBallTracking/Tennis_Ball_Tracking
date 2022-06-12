import numpy as np
import pandas as pd
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import torch
import torch.nn as nn
import detectron2.data.transforms as T
from skimage.feature import peak_local_max
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch.nn.functional as F

visibilities = [1,2,3]   # 1=flying ball on complex background to make it nearly invisible, 2=occuluded, 3=visible, 0=out of view
DT_THRESHOLD = 0.98
DT_WEIGHTS = 'soccer-ball-20220607-2034.pth'
BallNet_WEIGHTS = 'Soccer_BallNet_v4.pth'
images_folder = './images5/'
DEBUG = False

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DT_THRESHOLD # Set threshold for this model
cfg.MODEL.DEVICE = 'cuda'  # cuda or cpu
ball_detector1 = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(ball_detector1).load(DT_WEIGHTS)
ball_detector1.eval()

def extract_feature(frame):
    inputs = []
    original_height, original_width = frame.shape[:2]
    img = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    ).get_transform(frame).apply_image(frame)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs.append({"image": img, "height": original_height, "width": original_width})
    images = ball_detector1.preprocess_image(inputs)
    with torch.no_grad():
        features = ball_detector1.backbone(images.tensor)  # set of cnn features
        features_ = features['p2']
    height, width = features_.shape[2:4]
    features_ = features_.reshape(256, height, width)
    features_ = features_.cpu().numpy()
    return features_

device = torch.device("cuda")
class BallNet(nn.Module):
    def __init__(self):
        super(BallNet, self).__init__()
        self.conv1 = nn.Conv2d(3 * 256, 768, 3, padding=1)
        self.conv2 = nn.Conv2d(768, 2, 1)

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return out

model = BallNet()
model.to(device)
model = torch.load(BallNet_WEIGHTS)
model.eval()

def ball_detector2(tm1, t, tp1, threshold=0.0):
    features = []
    feature_ = extract_feature(tm1)
    original_height, original_width = tm1.shape[:2]
    height, width = feature_.shape[1:3]
    features.append(feature_)
    feature_ = extract_feature(t)
    features.append(feature_)
    feature_ = extract_feature(tp1)
    features.append(feature_)
    features = np.array(features).reshape(1, 3 * 256, height, width)
    out = model(torch.from_numpy(features).to(device))
    mask = (torch.argmax(out[0], dim=0)).detach().cpu().numpy().astype(np.uint8) * 255
    output = out.detach().cpu().numpy()[0,1]
    peaks = peak_local_max(output, min_distance=20, num_peaks=1)
    result = np.array([(x * original_height / height, y * original_height / height, output[y,x]) for y,x in peaks]).reshape(-1, 3)
    result = result[result[:,2]>threshold]
    if DEBUG:
        for y, x, in peaks:
            cv2.circle(mask, (x,y), 10, 255, 1)
        cv2.imshow('mask', mask)
    return result

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
        max_index = ball_gt.index.max()
        #cap = cv2.VideoCapture('../m-%03d.mp4' % MATCH_ID)
        for index, row in ball_gt[ball_gt.visible.isin(visibilities)].iterrows():
            # cap.set(0, row.time_ms - 80)
            # ret, tm1 = cap.read()
            # cap.set(0, row.time_ms)
            # ret, t = cap.read()
            # cap.set(0, row.time_ms + 80)
            # ret, tp1 = cap.read()
            tm1 = cv2.imread(images_folder + 'm-%03d-%08d.jpg' % (MATCH_ID, ball_gt.time_ms.loc[max(0, index-2)]))
            t = cv2.imread(images_folder + 'm-%03d-%08d.jpg' % (MATCH_ID, row.time_ms))
            tp1 = cv2.imread(images_folder + 'm-%03d-%08d.jpg' % (MATCH_ID, ball_gt.time_ms.loc[min(max_index, index+2)]))

            ball_centers = ball_detector2(tm1, t, tp1)[:,0:2]
            if ball_centers.size > 0:
                if DEBUG:
                    rounded_centers = np.round(ball_centers).astype(int)
                    [cv2.circle(t, tuple(c[0:2]), 15, (0, 255, 255), 2) for c in rounded_centers]
                distances = np.linalg.norm(ball_centers - [row.x, row.y], axis=1)
                if np.min(distances) < distance_threshold[MATCH_ID]:
                    correct += 1
                fp += (np.sum(distances >= distance_threshold[MATCH_ID]))
            else:
                fn += 1

            if DEBUG:
                cv2.imshow('frame', t)
                cv2.waitKey(int(1000/FPS))
    print('%d,%d,%d,%d,acc=%0.4f'%(match_id,correct,fp,fn,correct/(correct+fp+fn)))