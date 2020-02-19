from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms # needed for ImageFolder pre-processing

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

IMAGE_SIZE = 416

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_path", type=str, default="data/gate/classes.names", help="path to class label file")
    parser.add_argument("--model_def", type=str, default="config/yolov3-gate.cfg", help="path to model definition file")
    parser.add_argument("--checkpoint_model", type=str, default='', help="path to checkpoint model")
    parser.add_argument("--conf_thres", type=float, default=0.75, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--img_path", type=str, default='', help="path to image to detect live")

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.checkpoint_model:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.checkpoint_model))
    elif opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)

    model.eval()

    classes = load_classes(opt.class_path)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # receive an Image Tensor
    # pre-process according to ImageFolder class given to DataLoader

    #img = transforms.ToTensor()(img)
    img = transforms.ToTensor()(Image.open(opt.img_path))
    img, _ = pad_to_square(img, 0)
    img = resize(img, opt.img_size)
    img = Variable(img.type(Tensor))
    img = img.unsqueeze(0)

    with torch.no_grad():
        detections = model(img)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    detections = detections[0]
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    np_img = np.array(Image.open(opt.img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(np_img)

    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()
    