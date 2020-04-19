''' Benchmark that will is trained only on simulation data, and tested on real gate data.

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


#### CONSTANT VARIABLES ####
# File to store the evauluation of each epoch for certain training/validation size set
EVAL_RESULTS_FILE = 'benchmark_map.txt'
# File to store max mAP from using certain training/validation size set
FINAL_RESULTS_FILE = 'benchmark_results.txt'
# The split for the training and validation data
TRAIN_VAL_SPLIT = 0.8

def readInput():
    """Parse user input from the command line."""
    global DATA_DIR
    global IMG_EXT
    global DATA_CONFIG
    global MODEL_DEF
    global EPOCHS
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
    parser.add_argument("batch_size",
                        type=int,
                        help="batch size. For example 4.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    IMG_EXT = args.image_ext
    DATA_CONFIG = args.data_config
    MODEL_DEF = args.model_def
    EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size

def getAllImgs():
    # get all images into a list and shuffle
    all_imgs = os.listdir(os.path.join(DATA_DIR, "labels"))
    all_imgs = [img.split('.')[0] for img in all_imgs]
    random.shuffle(all_imgs)
    return all_imgs

def createTrainValSplit(all_imgs):
    # get random subset of images of size num_imgs, split 80/20 train/valid
    train_size = int(len(all_imgs) * TRAIN_VAL_SPLIT)
    train_imgs = all_imgs[:train_size]
    val_imgs = all_imgs[train_size:]
    
    # write training and valid images into txt files
    print("number of training images: {}".format(len(train_imgs)))
    print("number of validation images: {}".format(len(val_imgs)))

    # write training and valid images into txt files
    with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
        for img in train_imgs:
            f.write(os.path.join(DATA_DIR, "images", "") + img + IMG_EXT +"\n")

    with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
        for img in val_imgs:
            f.write(os.path.join(DATA_DIR, "images", "") + img + IMG_EXT +"\n")

if __name__ == "__main__":
    # read command line input
    readInput()

    # # get all images into a list and shuffle
    # all_imgs = getAllImgs()

    # # create train validation split
    # createTrainValSplit(all_imgs)

    # # delete contents of checkpoints before training
    # shutil.rmtree('checkpoints/')
    # os.mkdir('checkpoints/')


    #### BEGIN TRAINING ####
    for epoch in range(0, EPOCHS+1, 5):
        if epoch == 0:
            continue
        print("Model was trained for {} epochs.".format(epoch))

        

    # # Step 1: train model
    # subprocess.run(['python3','-W','ignore','train.py','--model_def', MODEL_DEF,
    # '--data_config', DATA_CONFIG,'--epochs', str(EPOCHS),'--batch_size', BATCH_SIZE])

    # # Step 2: stop after certain number of epochs

    # # Step 3: test the model

    # # Step 4: save results to benchmark file
    
    # # Step 5: save checkpoint model

    # # repeat Step 1 until total number of epochs reached