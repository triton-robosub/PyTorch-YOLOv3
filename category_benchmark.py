import os
import random
import subprocess 
import shutil 
import argparse

#### CONSTANT VARIABLES ####
# File to store the evauluation of each epoch for certain training/validation size set
EVAL_RESULTS_FILE = 'benchmark_map.txt'
# File to store max mAP from using certain training/validation size set
FINAL_RESULTS_FILE = 'benchmark_results.txt'
# Directories to store model checkpoints
CHKPT_DIRS = ['close_checkpoints/', 'medium_checkpoints/', 'checkpoints/']


#### USER INPUT ####
# Read in path to data files
print("What is the path to the data folder that contains the images, labels, etc.? For example data/custom.")
DATA_DIR = input("Path: ") 

# Read in path to .data file
print("What is the path to the data file for this model? For example config/custom.data.")
DATA_CONFIG = input("Data file: ") 

# Read in path to .cfg file
print("What is the path to the config file for this model? For example config/yolov3-custom.cfg.")
MODEL_DEF = input("Config file: ") 

# Read in the batch size for this benchmark
print("What is the number of epochs for this benchmark? An example entry is 4.")
BATCH_SIZE = input("Batch size: ")

# Read in the number of close epochs to train on close images for this benchmark
print("What is the number of close epochs for this benchmark? An example entry is 10.")
CLOSE_EPOCHS = int(input("Close epochs: "))

# Read in the number of medium epochs to train on medium images for this benchmark
print("What is the number of medium epochs for this benchmark? An example entry is 10.")
MEDIUM_EPOCHS = int(input("Medium epochs: "))

# Read in the number of far epochs to train on far images for this benchmark
print("What is the number of far epochs for this benchmark? An example entry is 10.")
FAR_EPOCHS = int(input("Far epochs: "))



######## SET UP CHECKPOINT DIRECTORIES #########
# delete contents of all checkpoint directories before training
if os.path.isdir(CHKPT_DIRS[0]):
    shutil.rmtree(CHKPT_DIRS[0])

if os.path.isdir(CHKPT_DIRS[1]):
    shutil.rmtree(CHKPT_DIRS[1])

if os.path.isdir(CHKPT_DIRS[2]):
    shutil.rmtree(CHKPT_DIRS[2])



########## Use current model state to train on close imgs ###########
# get list of all img numbers in close category and shuffle
close_imgs = [img.split('.')[0] for img in os.listdir(os.path.join(DATA_DIR, 'labels/close', ""))]
random.shuffle(close_imgs)

# write training and valid images into txt files
print(f'number of close training images: {len(close_imgs)}, number of close valid images: {len(close_imgs)}')
print('Writing train.txt and valid.txt')
with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
    for img in close_imgs:
        f.write(os.path.join(DATA_DIR, "images/close", "") + img + ".jpg\n")

with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
    for img in close_imgs:
        f.write(os.path.join(DATA_DIR, "images/close", "") + img + ".jpg\n")

# train model
subprocess.run(['python3','-W','ignore','train.py','--model_def', MODEL_DEF,
'--data_config', DATA_CONFIG,'--epochs', str(CLOSE_EPOCHS),'--batch_size', BATCH_SIZE])

# move checkpoints for close data to own directory
shutil.move(CHKPT_DIRS[2], CHKPT_DIRS[0])



########## Use current model state to train on medium imgs ###########
# get list of all img numbers in medium category and shuffle
medium_imgs = [img.split('.')[0] for img in os.listdir(os.path.join(DATA_DIR, 'labels/medium', ""))]
random.shuffle(medium_imgs)

#write training and valid images into txt files
print(f'number of medium training images: {len(medium_imgs)}, number of medium valid images: {len(medium_imgs)}')
print('Writing train.txt and valid.txt')
with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
    for img in medium_imgs:
        f.write(os.path.join(DATA_DIR, "images/medium", "") + img + ".jpg\n")

with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
    for img in medium_imgs:
        f.write(os.path.join(DATA_DIR, "images/medium", "") + img + ".jpg\n")

# train model
subprocess.run(['python3','-W','ignore','train.py','--model_def', MODEL_DEF,
'--data_config', DATA_CONFIG,'--epochs', str(MEDIUM_EPOCHS),'--batch_size', BATCH_SIZE])

# move checkpoints for medium data to own directory
shutil.move(CHKPT_DIRS[2], CHKPT_DIRS[1])



########## Use current model state to train on far imgs ###########
# get list of all img numbers in far category and shuffle
far_imgs = [img.split('.')[0] for img in os.listdir(os.path.join(DATA_DIR, 'labels/far', ""))]
random.shuffle(far_imgs)

#write training and valid images into txt files
print(f'number of far training images: {len(far_imgs)}, number of far valid images: {len(far_imgs)}')
print('Writing train.txt and valid.txt')
with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
    for img in far_imgs:
        f.write(os.path.join(DATA_DIR, "images/far", "") + img + ".jpg\n")

with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
    for img in far_imgs:
        f.write(os.path.join(DATA_DIR, "images/far", "") + img + ".jpg\n")

# train model
subprocess.run(['python3','-W','ignore','train.py','--model_def', MODEL_DEF,
'--data_config', DATA_CONFIG,'--epochs', str(FAR_EPOCHS),'--batch_size', BATCH_SIZE])



################ TEST THE MODEL #####################

# remove final results file to have fresh results fike
os.remove(EVAL_RESULTS_FILE)

# Evaluate the model based on each category and its checkpoints
for category in CHKPT_DIRS:
    # test model for each epoch, record highest mAP and write to file. 
    metrics = []

    # get each checkpoint model in this category
    for model in os.listdir(category):
        weights_path = category + model

        # Evaluate using the checkpoint weights from this EPOCH
        subprocess.run(['python3', 'test.py', '--batch_size', BATCH_SIZE, '--model_def', MODEL_DEF, '--data_config', 
        DATA_CONFIG, '--weights_path', weights_path, '--class_path', os.path.join(DATA_DIR, "classes.names")])

        # Save the mAP after the evaluation for each epoch
        with open(EVAL_RESULTS_FILE,'r+') as f:
            mAP = f.read()
        metrics.append(mAP)

    # Take the max mAP for this number of training/validation images and save it into the results file
    max_mAP = max(metrics)
    results = open(FINAL_RESULTS_FILE, 'a+')
    results.write(category + ' : ' + str(max_mAP) + '\n')
    results.close()
