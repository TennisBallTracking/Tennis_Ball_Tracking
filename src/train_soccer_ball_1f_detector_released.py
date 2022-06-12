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

OUTPUT_FOLDER = './out_20220606'

def get_ball_dicts(img_dir, split):
    ball_gt_df = pd.read_csv('ball-gt-v4.csv')
    if split == "train":
        ball_gt_df = ball_gt_df[:4002]
    else:
        ball_gt_df = ball_gt_df[4002:]

    dataset_dicts = []
    for idx, row in ball_gt_df.iterrows():
        record = {}

        filename = os.path.join(img_dir, row.filename)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = [{"bbox": [row.x1, row.y1, row.x2, row.y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,}]
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("ball_" + d, lambda d=d: get_ball_dicts("./", d))
    MetadataCatalog.get("ball_" + d).set(thing_classes=["ball"])
ball_metadata = MetadataCatalog.get("ball_train")

# dataset_dicts = get_ball_dicts("./", "train")
# for d in random.sample(dataset_dicts, 30):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=ball_metadata, scale=1.0)
#     out = visualizer.draw_dataset_dict(d)
#     print(d["file_name"], d["annotations"])
#     cv2.imshow("test", cv2.resize(out.get_image()[:, :, ::-1], (1920,1080)))
#     cv2.waitKey(0)

cfg = get_cfg()
cfg.OUTPUT_DIR = OUTPUT_FOLDER
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ball_train",)
cfg.DATASETS.TEST = ("ball_val",)
# cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = 'soccer-ball-202110070754.pth' # Alternatively, initialize from a trained model
cfg.SOLVER.IMS_PER_BATCH = 3  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 60000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.5, 0.5]
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

cfg = get_cfg()   # get a fresh new config
cfg.OUTPUT_DIR = OUTPUT_FOLDER
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
predictor = DefaultPredictor(cfg)

dataset_dicts = get_ball_dicts("./", "val")
correct, fp, fn = 0, 0, 0
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    ball_boxes = outputs['instances'][outputs['instances'].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
    print(ball_boxes)
    gt = d["annotations"][0]["bbox"]
    if ball_boxes.size > 0:
        ball_centers = np.concatenate(
            [(ball_boxes[:, 0:1] + ball_boxes[:, 2:3]) / 2, (ball_boxes[:, 1:2] + ball_boxes[:, 3:4]) / 2], axis=1)
        gt_center = np.array([(gt[0]+gt[2])/2, (gt[1]+gt[3])/2])
        distances = np.linalg.norm(ball_centers - gt_center, axis=1)
        if np.min(distances) < 10:
            correct += 1
        fp += np.sum(distances >= 10)
    else:
        fn += 1
    print('%d,%d,%d,%0.3f'%(correct, fp, fn, correct/(correct+fp+fn)))



