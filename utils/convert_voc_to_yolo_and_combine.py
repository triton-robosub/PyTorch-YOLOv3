"""
Recursively combines all images and labels below root directory into 
one images and one labels directory in root directory. Also converts
PascalVOC labels into YOLOv3 labels. Creates a classes.names file for
classes passed in.

Usage: python ~/Desktop/PyTorch-YOLOv3/utils/convert_voc_to_yolo_and_combine.py [root_data_directory] --class_names class1 class2 ...

Author: Imran Matin
Credits: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
"""
import shutil
import os
import argparse

# imports below are requred for convert_annotation()
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join
from pathlib import Path
import sys

# uncomment and comment relevant lines in parseCmdLine() if wnat to hardcode classes
# CLASSES = ["bp", "lp"]

def parseCmdLine():
    global CLASSES
    global ROOT_DIR
    global ROOT_IMG_DIR
    global ROOT_XML_LABELS_DIR
    global ROOT_LABELS_DIR

    # Construct the argument parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data",
                        type=str,
                        help="path to the data folder")
    parser.add_argument('--class_names',
                        nargs='+',
                        help="List of classes for the ROOT directory images.")
    args = parser.parse_args()

    # get classes for the data
    CLASSES = args.class_names
    
    # Create paths to new image and labels directories
    ROOT_DIR = args.path_to_data
    ROOT_IMG_DIR = os.path.join(ROOT_DIR, "images")
    ROOT_XML_LABELS_DIR = os.path.join(ROOT_DIR, "xml_labels")
    ROOT_LABELS_DIR = os.path.join(ROOT_DIR, "labels")

def createFinalDirs():
    # remove the images and labels directories if they already exist
    # TODO: CAUTION!!!!
    if os.path.exists(ROOT_IMG_DIR):
        shutil.rmtree(ROOT_IMG_DIR)
    if os.path.exists(ROOT_XML_LABELS_DIR):
        shutil.rmtree(ROOT_XML_LABELS_DIR)
    if os.path.exists(ROOT_LABELS_DIR):
        shutil.rmtree(ROOT_LABELS_DIR)
    
    # make new image and labels directories
    os.mkdir(ROOT_IMG_DIR)
    os.mkdir(ROOT_XML_LABELS_DIR)
    os.mkdir(ROOT_LABELS_DIR)

def copyFiles(currDir):
    """Recursive function to iterate down file hiearchy or copy base case files to ROOT directory."""

    # check if this is a file containing raw images and VOC labels
    for filename in os.listdir(currDir):
        if filename.endswith(".xml") or filename.endswith(".png"):
            # sort the files in the into xml and images
            images, xml_labels = sortFiles(currDir)
            
            # move the images and xml labels to root directory images and labels folders
            moveFiles(currDir, images, xml_labels)
            return

    # get a list of all the subdirectories for the current directory
    for subdir in os.listdir(currDir):
        subdir = os.path.join(currDir, subdir)
        # if images or labels directory, then skip
        if (subdir == ROOT_IMG_DIR or 
            subdir == ROOT_XML_LABELS_DIR or
            "classes.names" in subdir):
            continue
        copyFiles(subdir)

def sortFiles(subdir):
    """Seperates .xml and .png files in directory into two lists."""
    images = []
    xml_labels = []

    for f in os.listdir(subdir):
        if ".png" in f:
            images.append(f)
        elif ".xml" in f:
            xml_labels.append(f)

    return images, xml_labels

def moveFiles(subdir, images, xml_labels):
    for img, xml_label in zip(images, xml_labels):
            # create path to image and label
            img_path = os.path.join(subdir, img)
            xml_label_path = os.path.join(subdir, xml_label)

            # copy image and label to respective root directories
            shutil.copy(img_path, ROOT_IMG_DIR)
            shutil.copy(xml_label_path, ROOT_XML_LABELS_DIR)

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

def renameLabels():
    for label in os.listdir(ROOT_LABELS_DIR):
            old_label_path = os.path.join(ROOT_LABELS_DIR, label)
            new_label_path = os.path.join(ROOT_LABELS_DIR, label[6:])
            os.rename(old_label_path, new_label_path)

def write_class_names():
    """Write class names to the YOLOv3 directory.
    
    Args:
        classes: List of class names.
    """
    cpath = os.path.join(ROOT_DIR, "classes.names")
    with open(cpath, "w") as cfile:
        cfile.writelines("\n".join(CLASSES))

if __name__ == "__main__":
    # parse command line arguments
    parseCmdLine()

    # create ROOT directory images and labels dir
    createFinalDirs()

    # copy all images and labels of sub directories into ROOT images and labels directory
    copyFiles(ROOT_DIR)

    # convert to YOLOv3 format
    convert_annotation(ROOT_XML_LABELS_DIR, ROOT_LABELS_DIR)

    # remove the xml labels directory
    shutil.rmtree(ROOT_XML_LABELS_DIR)

    # remove rename labels to match images
    renameLabels()

    # write classes.names file
    write_class_names()
