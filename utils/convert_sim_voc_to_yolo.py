""" Convert PascalVOC formatted data (.xml) to YOLOv3 data format (.txt).

Usage: `python convert_sim_voc_to_yolo.py path_to_data`

path_to_data should be a folder that has all images and corresponding .xml files.
This script will seperate all images and label files, and then convert PascalVOC
formatted data (.xml) to YOLOv3 data format (.txt). It will also create a classes.names
file for the converted data.

Author: Imran Matin
Credits: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from pathlib import Path
import sys
import getopt
import argparse

# new directories to make
IMAGES = "images"
XML_LABELS = "xml_labels"
TXT_LABELS = "txt_labels"
CLASSES = ["p,lp", "p,bp"]


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(voc_path, yolo_path):
    """Convert PascalVOC annotations to YOLOv3.
    
    Args:
        voc_path: Path to the PascalVOC directory.
        yolo_path: Path to the YOLOv3 directory.
    """
    if not isinstance(voc_path, Path):
        voc_path = Path(voc_path)
    if not isinstance(yolo_path, Path):
        yolo_path = Path(yolo_path)
    annpath = voc_path
    outpath = yolo_path

    outpath.mkdir(parents=True, exist_ok=True)

    for in_fname in annpath.iterdir():
        out_fname = outpath / (in_fname.name.split(".")[0] + ".txt")
        with open(in_fname) as in_file, open(out_fname, "w") as out_file:

            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)

            for obj in root.iter("object"):
                # no difficult attribute in sim data
                # difficult = obj.find("difficult").text
                cls = obj.find("name").text
                # if cls not in classes or int(difficult) == 1:
                if cls not in CLASSES:
                    continue
                cls_id = CLASSES.index(cls)
                xmlbox = obj.find("bndbox")
                b = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text),
                )
                bb = convert((w, h), b)
                out_file.write(
                    str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n"
                )

def write_class_names():
    """Write class names to the YOLOv3 directory.
    
    Args:
        classes: List of class names.
    """
    cpath = "classes.names"
    with open(cpath, "w") as cfile:
        cfile.writelines("\n".join(CLASSES))

def sort_files():
    # make the new directories
    os.mkdir(IMAGES)
    os.mkdir(XML_LABELS)
    os.mkdir(TXT_LABELS)

    # move files to correct directory
    for f in os.listdir("."):
        if ".png" in f:
            os.rename(f, os.path.join(IMAGES, f))
        elif ".xml" in f:
            # removes label_ in front of .xml files
            new_f = f[6:]
            os.rename(f, os.path.join(XML_LABELS, new_f))

if __name__ == "__main__":
    # Construct the argument parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data", type=str,
                        help="path to the data folder")
    args = parser.parse_args()

    # enter that directory
    os.chdir(args.path_to_data)

    # move images and label files to correct folders
    sort_files()
    # convert to YOLOv3 format
    convert_annotation(XML_LABELS, TXT_LABELS)
    # write classes.names file
    write_class_names()