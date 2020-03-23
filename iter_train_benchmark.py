import os
import random
import subprocess 
import shutil 

#### CONSTANT VARIABLES ####
# upperbound for number of total training labels, currently 1812
MAX_IMGS = 1900 
# anything less will result in errors during validation stage.
START_IMGS = 700 
# Number of images to increase curren number of images used for training and validation split
STEP_SIZE = 100
# File to store the evauluation of each epoch for certain training/validation size set
EVAL_RESULTS_FILE = 'benchmark_map.txt'
# File to store max mAP from using certain training/validation size set
FINAL_RESULTS_FILE = 'benchmark_results.txt'



#### USER INPUT ####
# Read in path to data files
print("What is the path to the data folder that contains the images, labels, etc.? For example data/custom.")
DATA_DIR = input("Path: ") 

# Read in path to .data file
print("What is the path to the config file for this model? For example config/custom.data.")
DATA_CONFIG = input("Data file: ") 

# Read in path to .cfg file
print("What is the path to the config file for this model? For example config/yolov3-custom.cfg.")
MODEL_DEF = input("Config file: ") 

# Read in the number of epochs for this benchmark
print("What is the number of epochs for this benchmark? An example entry is 50.")
EPOCHS = int(input("Epochs: "))

# Read in the batch size for this benchmark
print("What is the number of epochs for this benchmark? An example entry is 4.")
BATCH_SIZE = input("Batch size: ")



#### BEGIN TRAINING ####
# get all images into a list and shuffle
all_imgs = os.listdir(os.path.join(DATA_DIR, "labels"))
all_imgs = [img.split('.')[0] for img in all_imgs]
random.shuffle(all_imgs)

# benchmark based on different number of images used for training/validation
for num_imgs in range(START_IMGS, MAX_IMGS, STEP_SIZE):
    print(num_imgs)

    # always want to include same images as previous num_imgs
    # ex. 800 images has same 700 + 100 additional random images, for consistency
    cur_subset = all_imgs[:num_imgs]
    random.shuffle(cur_subset)
    # get random subset of images of size num_imgs, split 80/20 train/valid
    train_size = int(num_imgs * 0.8)
    train_img = cur_subset[:train_size]
    valid_img = cur_subset[train_size:num_imgs]

    # write training and valid images into txt files
    print(f'number of training images: {len(train_img)}, number of valid images: {len(valid_img)}')
    print('Writing train.txt and valid.txt')
    with open(os.path.join(DATA_DIR, "train.txt"),'w') as f:
        for img in train_img:
            f.write(os.path.join(DATA_DIR, "images", "") + img + ".jpg\n")

    with open(os.path.join(DATA_DIR, "valid.txt"),'w') as f:
        for img in valid_img:
            f.write(os.path.join(DATA_DIR, "images", "") + img + ".jpg\n")
    
    # delete contents of checkpoints before training
    shutil.rmtree('checkpoints/')
    os.mkdir('checkpoints/')

    # train model
    subprocess.run(['python3','-W','ignore','train.py','--model_def', MODEL_DEF,
    '--data_config', DATA_CONFIG,'--epochs', str(EPOCHS),'--batch_size', BATCH_SIZE])

    #test model for each epoch, record highest mAP and write to file. 
    metrics = []

    for i in range(0, EPOCHS + 1, 1):
        # Save the weights for this EPOCH
        if i == EPOCHS:
            i = EPOCHS - 1
        weights_path = f'checkpoints/yolov3_ckpt_{i}.pth'

        # Evaluate using the checkpoint weights from this EPOCH
        subprocess.run(['python3', 'test.py', '--batch_size', BATCH_SIZE, '--model_def', MODEL_DEF,
        '--data_config', DATA_CONFIG, '--weights_path', weights_path, '--class_path', os.path.join(DATA_DIR, "classes.names")])

        # Save the mAP after the evaluation for each epoch
        with open(EVAL_RESULTS_FILE,'r+') as f:
            mAP = f.read()
        metrics.append(mAP)

    # Take the max mAP for this number of training/validation images and save it into the results file
    max_mAP = max(metrics)
    results = open(FINAL_RESULTS_FILE, 'a+')
    results.write(str(num_imgs) + 'images :' + str(max_mAP) + '\n')
    results.close()
