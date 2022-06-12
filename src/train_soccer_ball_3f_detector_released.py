import pandas as pd
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# # prepare training images
# train_df = pd.read_csv('ball_net_train_v2.csv')
# for index, row in train_df.iterrows():
#     cap = cv2.VideoCapture('../m-%03d.mp4' % row.match_id)
#     for t in [row.time_ms, row.tm1, row.tp1]:
#         cap.set(0, t)
#         ret, frame = cap.read()
#         cv2.imwrite('images4/m-%03d-%08d.jpg'%(row.match_id, t), frame)
#         print('images4/m-%03d-%08d.jpg'%(row.match_id, t))

image_folder = 'images4/'
DT_THRESHOLD = 0.98
DT_WEIGHTS = 'soccer-ball-20220607-2034.pth'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DT_THRESHOLD # Set threshold for this model
our_predictor = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(our_predictor).load(DT_WEIGHTS)
our_predictor.eval()

def extract_feature(frame):
    inputs = []
    original_height, original_width = frame.shape[:2]
    img = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    ).get_transform(frame).apply_image(frame)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs.append({"image": img, "height": original_height, "width": original_width})
    images = our_predictor.preprocess_image(inputs)
    with torch.no_grad():
        features = our_predictor.backbone(images.tensor)  # set of cnn features
        features_ = features['p2']
    height, width = features_.shape[2:4]
    features_ = features_.reshape(256, height, width)
    features_ = features_.cpu().numpy()
    return features_

class BallDataset(object):
    def __init__(self, ball_df, img_folder='images4/'):
        self.ball_df = ball_df
        self.img_folder = img_folder

    def __getitem__(self, idx):
        row = self.ball_df.iloc[idx].to_numpy()
        match_id,time_ms,x,y,tm1,tp1 = row[0:6]
        features = []
        for t in [tm1, time_ms, tp1]:
            img = cv2.imread(self.img_folder+'m-%03d-%08d.jpg'%(match_id,t))
            original_height, original_width = img.shape[:2]
            features_ = extract_feature(img)
            height, width = features_.shape[1:3]
            features.append(features_)
        features = np.array(features).reshape(3 * 256, height, width)
        target = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(target, tuple(np.round(np.array([x, y]) * height / original_height).astype(np.int)), 3, 1, -1)
        target = torch.as_tensor(target.astype(np.longlong))
        return features, target

    def __len__(self):
        return self.ball_df.shape[0]

train_df = pd.read_csv('ball_net_train_v2.csv')
train_dataset = BallDataset(train_df)

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

def train_ballnet(model, model_saved):
    lr = 1e-4
    loss_function = nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return torch.mean((preds == yb).float())

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    epochs = 5
    # model = torch.load('BallNet_v7.pth')  # Transfer learning
    model.train()
    for i in range(epochs):
        j = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = loss_function(output, labels, weight=torch.tensor([1.0, 100.0]).to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            j += 1
            print(i,j)

    torch.save(model, model_saved)

train_ballnet(model, 'Soccer_BallNet_v4.pth')