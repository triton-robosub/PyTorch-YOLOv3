from __future__ import division

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-gate.cfg", help="path to model definition file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default='', help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.checkpoint_model:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.checkpoint_model))

    model = model.to('cpu')
    torch.save(model.state_dict(), "cpu_weights.pth")