import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import os.path
import time, sys
import cv2

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

def load(path):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

config_file = "../configs/e2e_faster_rcnn_R_101_FPN_1x.yaml"
#config_file = "../configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

root = "/home/jiasheng/maskrcnn-benchmark/datasets/neo/"    
dataset_name = "Neovision2-Training-Heli-047"
width = 960
height = 540
video = cv2.VideoWriter(dataset_name+"-predictions_24.avi", 0, 24, (width,height))
for img in sorted(os.listdir(os.path.join(root, dataset_name))):
    path = os.path.join(root + dataset_name,img)
    image = load(path)
    print(path)
    predictions = coco_demo.run_on_opencv_image(image)
    frame = cv2.resize(predictions,(width, height))    
    video.write(frame)

cv2.destroyAllWindows()
video.release()
# image = load('/home/jiasheng/maskrcnn-benchmark/datasets/neo/Neovision2-Training-Heli-013/000110.png')
# predictions = coco_demo.run_on_opencv_image(image)
# plt.figure()
# plt.imshow(predictions) 
# plt.show()  