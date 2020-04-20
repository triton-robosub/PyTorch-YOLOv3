''' Benchmark that will is trained only on simulation data, and tested on real gate data.

Usage: python sim_data_benchmark.py data/ .png config/sim_data.data config/yolov3-sim_data.cfg 50 5 4

The benchmark will train a YOLOv3 custom model on only simulation data for the gate captured
in the Unity simulation. It will test on a test set composed of real gate data. The process
consists training for a certain number of epochs, and then testing on the test set. This process
repeats until overfitting is realized. A results log will contain a checkpoint model and a testing
benchmark score each time the model tests on the test set.

Author: Imran Matin
'''
import argparse
import shutil
import os
import random
import subprocess
import json
from datetime import datetime
import getpass

#### CONSTANT VARIABLES ####
# File to store the evauluation of each epoch for certain training/validation size set
EVAL_RESULTS_FILE = 'benchmark_map.txt'
# File to store max mAP from using certain training/validation size set
FINAL_RESULTS_FILE = 'benchmark_results.json'
# The split for the training and validation data
TRAIN_VAL_SPLIT = 0.95

def readInput():
    """Parse user input from the command line."""
    global DATA_DIR
    global IMG_EXT
    global DATA_CONFIG
    global MODEL_DEF
    global EPOCHS
    global TEST_INTERVAL
    global BATCH_SIZE


    # Construct the argument parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        type=str,
                        help="path to the data folder. For example data/custom.")
    parser.add_argument("image_ext",
                        type=str,
                        help="file type for images. For example .png")
    parser.add_argument("data_config",
                        type=str,
                        help="path to the data file. For example config/custom.data.")
    parser.add_argument("model_def",
                        type=str,
                        help="path to the config file. For example config/yolov3-custom.cfg.")
    parser.add_argument("num_epochs",
                        type=int,
                        help="number of epochs. For example 50.")
    parser.add_argument("test_interval",
                        type=int,
                        help="every interval to test the model. For example 5.")
    parser.add_argument("batch_size",
                        type=int,
                        help="batch size. For example 4.")

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    IMG_EXT = args.image_ext
    DATA_CONFIG = args.data_config
    MODEL_DEF = args.model_def
    EPOCHS = args.num_epochs
    TEST_INTERVAL = args.test_interval
    BATCH_SIZE = args.batch_size

def getAllImgs():
    """Read all images into a list and then shuffle the list."""
    # get all images into a list and shuffle
    all_imgs = os.listdir(os.path.join(DATA_DIR, "labels"))
    all_imgs = [img.split('.')[0] for img in all_imgs]
    random.shuffle(all_imgs)

    return all_imgs

def createTrainValSplit(all_imgs):
    """Split the images into train and valid and write the names to files."""

    # get random subset of images of size num_imgs, split 80/20 train/valid
    train_size = int(len(all_imgs) * TRAIN_VAL_SPLIT)
    train_imgs = all_imgs[:train_size]
    val_imgs = all_imgs[train_size:]

    # write training and valid images into txt files
    with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
        for img in train_imgs:
            f.write(os.path.join(DATA_DIR, "images", "") + img + IMG_EXT +"\n")

    with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
        for img in val_imgs:
            f.write(os.path.join(DATA_DIR, "images", "") + img + IMG_EXT +"\n")
    
    return train_imgs, val_imgs

def trainModel():
    """Train the model for specified Epochs."""
    subprocess.run(['python3','-W','ignore','train.py',
    '--model_def', MODEL_DEF,
    '--data_config', DATA_CONFIG,
    '--epochs', str(EPOCHS),
    '--batch_size', str(BATCH_SIZE)])

def testModel():
    """Test the model for every specified number of epochs trained and save the mAP."""
    # test the model after every 5 epochs of training
    # range will skip last epoch value if not included because 
    for ckpt in range(0, EPOCHS+1, TEST_INTERVAL):
        # model only has checkpoints from [0, EPOCHS-1]
        if ckpt == EPOCHS:
            ckpt = EPOCHS - 1

        weights_path = f'checkpoints/yolov3_ckpt_{ckpt}.pth'

        subprocess.run(['python3', 'test.py',
        '--model_def', MODEL_DEF,
        '--data_config', DATA_CONFIG,
        '--batch_size', str(BATCH_SIZE),
        '--weights_path', weights_path, 
        '--class_path', os.path.join(DATA_DIR, "classes.names")])

        # Save the mAP after the evaluation for each epoch
        # need to have this file created before training
        with open(EVAL_RESULTS_FILE,'r+') as f:
            mAP = f.read()
        metrics["results"][weights_path] = mAP

def writeResults(metrics):
    """Write the results of the benchmark to the file."""
    # write training results and information to file
    with open(FINAL_RESULTS_FILE, 'w+') as outfile:
        json.dump(metrics, outfile)

if __name__ == "__main__":
    # read command line input
    readInput()

    # get all images into a list and shuffle
    all_imgs = getAllImgs()

    # create train validation split
    train_imgs, val_imgs = createTrainValSplit(all_imgs)

    #### BEGIN TRAINING ####
    metrics = { "Model Trainer": getpass.getuser(), 
                "Date": str(datetime.now()),
                "Epochs Trained For": EPOCHS,
                "Test Interval": TEST_INTERVAL,
                "Batch Size": BATCH_SIZE,
                "Image Type": IMG_EXT,
                "Number of Training Images": len(train_imgs),
                "Number of Validation Images": len(val_imgs),
                "results": {}
                }

    # delete contents of checkpoints before training
    shutil.rmtree('checkpoints/')
    os.mkdir('checkpoints/')
    # Train model
    trainModel()
    # Test model
    testModel()
    # write model training results to file
    writeResults(metrics)