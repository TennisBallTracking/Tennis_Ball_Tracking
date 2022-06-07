# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm as tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.modeling import RPN_HEAD_REGISTRY
from detectron2.layers import ShapeSpec
import torch
import os
from __future__ import print_function, division
torch.multiprocessing.set_start_method('spawn')
from numpy import unravel_index
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

# ConvNet, perform one 3*3 conv followed by a 1*1 conv
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1280, 768, kernel_size=3, padding=1) # the output size can be tuned
        self.conv2 = nn.Conv2d(768, 1, kernel_size=1)
        self.relu = nn.ReLU() 
        
    def forward(self, features: List[torch.Tensor], batch_size):
        h, w = features.shape[3], features.shape[4]
        input = features.reshape(batch_size, 1280, h, w).to(device)
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# Redefine RPN
objectness = None
@RPN_HEAD_REGISTRY.register()
class MyRPN(detectron2.modeling.proposal_generator.rpn.StandardRPNHead):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

    def forward(self, features):
        global objectness  # to store the objectness_logits in my code
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            objectness = t  # this is my code
            pred_anchor_deltas.append(self.anchor_deltas(t))
            # print('my rpn is called')
        return pred_objectness_logits, pred_anchor_deltas


# DataLoader
class BallDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. "match_name, time, xc, yc, is_bounce"
            root_dir (string): Directory with all the images/video.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        time = int(self.landmarks_frame.iloc[idx, 1])
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        inputs = []
        match_id = int(self.landmarks_frame.iloc[idx, 0])
        for offset in [-80, -40, 0, 40, 80]:
            frame = cv2.imread('../train_imgs/%d_%d.jpg'%(match_id, time + offset))
            H, W, _ = frame.shape
            image = torch.as_tensor(frame.astype('float32').transpose(2, 0, 1))
            inputs.append({'image': image, 'height': H, 'width': W})
        images = detector.preprocess_image(inputs)
        with torch.no_grad():
            features = detector.backbone(images.tensor)  # set of cnn features
            proposals, _ = detector.proposal_generator(images, features, None)
            feature_maps = objectness[:, :, :H//4, :W//4]
        fh, fw = feature_maps.shape[-2:]
        target = np.zeros((fh, fw), dtype=np.uint8)
        temp_x, temp_y = int(landmarks[0] / 4), int(landmarks[1] / 4)
        cv2.circle(target, (temp_x, temp_y), 2, 1, -1)
        target = torch.tensor(target, dtype=torch.long).to(device)
        sample = {'feature_maps': feature_maps, 'landmarks': target}
        return sample


# Train ConvNet
CSV_FILE = '../csv_files/ball_train_small.csv'
MODEL_FILE = '../models/spatial_temporal_model/ball_model_convnet_5.pth'
ball_dataset = BallDataset(csv_file=CSV_FILE, root_dir='videos/')
batch_size = 2
dataloader = DataLoader(ball_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
model = ConvNet().to(device)
loss_function = nn.CrossEntropyLoss(torch.FloatTensor([1.0, 1.0])).to(device)
optimizer = optim.Adadelta(params=model.parameters(), lr=1.0)

cfg = get_cfg()
cfg.merge_from_file('./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball)
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32]]
cfg.MODEL.RPN.IN_FEATURES = ['p2']
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.HEAD_NAME = 'MyRPN'
detector = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(detector).load('../models/single_image_models/model_final.pth')
detector.eval()

for epoch in tqdm(range(2)):
    total_train = 0
    correct_train = 0
    for i_batch, sample_batched in enumerate(dataloader):
        feature_maps = sample_batched['feature_maps']
        landmarks = sample_batched['landmarks']
        fh, fw = feature_maps.shape[-2:]
        landmarks = torch.reshape(landmarks, (batch_size, fh * fw))
        optimizer.zero_grad()
        prediction = model(feature_maps, batch_size)
        prediction = torch.reshape(prediction, (batch_size, fh * fw))
        # Perform random sampling
        np_landmarks = landmarks.cpu().detach().numpy()
        landmarks_index = []
        pred_index = []
        for i in range(batch_size): # for every element in a batch
            num_foreground_pixels = np.count_nonzero(np_landmarks[i] > 0) # number of ball pixels
            foreground_index = np.where(np_landmarks[i] > 0)[0] # index of ball pixels
            background_index = np.random.choice(np.where(np_landmarks[i] == 0)[0], 300 - num_foreground_pixels,
                                                replace=False) # index of background pixels after down sampling
            total_index = np.append(foreground_index, background_index, 0)
            landmarks_index.append(total_index)
            pred_index.append(total_index)
        landmarks_index = torch.tensor(landmarks_index).to(device)
        pred_index = torch.tensor(pred_index).to(device)
        new_landmarks = torch.gather(landmarks, 1, landmarks_index)
        new_pred = torch.gather(prediction, 1, pred_index)
        loss = F.binary_cross_entropy_with_logits(new_pred, new_landmarks.to(torch.float32), reduction='sum')
        loss.backward()
        optimizer.step()
        if i_batch % 10 == 0:
            print('Epoch {}, train Loss: {:.3f}'.format(epoch, loss.item()))
        if i_batch % 100 == 0:
            print('Save a model!')
            torch.save((model.state_dict()), MODEL_FILE)
    torch.save((model.state_dict()), MODEL_FILE)


# Validation
cfg = get_cfg()
cfg.merge_from_file('./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball)
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32]]
cfg.MODEL.RPN.IN_FEATURES = ['p2']
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.HEAD_NAME = 'MyRPN'
detector = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(detector).load('../models/single_image_models/model_final.pth')
detector.eval()

model = ConvNet()
model.load_state_dict(torch.load('../models/spatial_temporal_model/ball_model_convnet_5.pth', map_location='cuda'))
model.to(device)
model.eval()

MATCH_ID = 20
df_test = pd.read_csv('../csv_files/m-%03d-ball-gt.csv'%(MATCH_ID))
result = []
for index, row in tqdm(df_test.iterrows()):
    x_gt, y_gt = row.x, row.y
    match_id = int(row.match_id)
    time_ms = int(row.time_ms)
    inputs = []
    for offset in [-80, -40, 0, 40, 80]:
        frame = cv2.imread('../train_imgs/%d_%d.jpg' % (match_id, time_ms + offset))
        H, W, _ = frame.shape
        image = torch.as_tensor(frame.astype('float32').transpose(2, 0, 1))
        inputs.append({'image': image, 'height': H, 'width': W})
    images = detector.preprocess_image(inputs)
    with torch.no_grad():
        features = detector.backbone(images.tensor)  # set of cnn features
        proposals, _ = detector.proposal_generator(images, features, None)
        feature_maps = objectness[:, :, :H//4, :W//4]
    fh, fw = feature_maps.shape[-2:]
    feature_maps = torch.reshape(feature_maps, (1, 5, 256, fh, fw))
    with torch.no_grad():
        pred = model(feature_maps, 1)
    
    # thresholding
    thred = pred.cpu().detach().numpy()[0]
    thred[thred > -20] = 255
    thred[thred <= -20] = 0
    binary_img = np.array(thred[0], dtype=np.uint8)
    
    # obtain contours
    cnts, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    if len(cnts) == 0:
        continue
    
    # check whether top 10 blobs has correct prediction
    true_score = float('-inf')
    max_score = float('-inf')
    x_final, y_final = -1, -1
    for cnt in cnts:
        final = np.zeros(binary_img.shape, np.uint8)
        cv2.drawContours(final, [cnt], -1, 255, cv2.FILLED)
        temp_pred = pred.cpu().detach().numpy()[0][0]
        temp_pred[final == 0] = float('-inf')
        cY, cX = unravel_index(temp_pred.argmax(), temp_pred.shape)
        x_pred, y_pred = cX * 4, cY * 4
        result.append([match_id, time_ms, row.ball_sn, x_pred, y_pred, temp_pred[cY, cX]])
    
result = pd.DataFrame(result, columns=['match_id', 'time_ms', 'ball_sn', 'x', 'y', 'score'])
result.to_csv('../csv_files/m-%03d-five.csv', index=False)
