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


import glob
import os

import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron2.')
parser.add_argument('--scene',type=int, default = 1, help='data scene number')
parser.add_argument('--direction',type=str, default ='front', help='front data for True and rear data for False')
args = parser.parse_args()


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

home_dir ='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data' 
data_dir = home_dir+'/scene'+str(args.scene)+'/'+args.direction
save_file_name = home_dir+'/object_detection/predictions-scene-'+str(args.scene)+'-'+args.direction+'.pickle'
data_path = os.path.join(data_dir,'*png')
files = glob.glob(data_path)

pred_classes = []
pred_boxes = []
image_name = []
for num, images in enumerate(files):
    image_name.append(images)
    im = cv2.imread(images)#, cv2.IMREAD_GRAYSCALE)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("prediction", out.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    pred_classes.append(outputs["instances"].pred_classes.to("cpu").numpy())
    pred_boxes.append(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    print(images)

#'''
import pickle
with open(save_file_name, 'wb') as f:
    pickle.dump([pred_classes, pred_boxes, image_name], f)


with open(save_file_name, 'rb') as f:
  var = pickle.load(f)

print(var)
#'''


