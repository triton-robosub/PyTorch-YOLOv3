import os
import random
import subprocess 
import shutil 
import argparse

# TMUX instructions
# ctrl-b then d to detach
# to get back, tmux attach

# INSTRUCTIONS
# Create empty directories for close_checkpoints, medium_checkpoints, checkpoints
# mkdir close_checkpoints
# mkdir medium_checkpoints
# mkdir checkpoints
# conda activate robosub
# python3 imran_benchmark.py --close_epochs NUM_CLOSE_EPOCHS --medium_epochs NUM_MEDIUM_EPOCHS --far_epochs NUM_FAR_EPOCHS


# Take number of epochs as command line args
parser = argparse.ArgumentParser()
parser.add_argument("--close_epochs", type=int, default=10, help="number of close epochs")
parser.add_argument("--medium_epochs", type=int, default=10, help="number of medium epochs")
parser.add_argument("--far_epochs", type=int, default=10, help="number of far epochs")
opt = parser.parse_args()
print(opt)

# Number of epochs to train model on close images
CLOSE_EPOCHS = opt.close_epochs
# Number of epochs to train model on medium images
MEDIUM_EPOCHS = opt.medium_epochs
# Number of epochs to train model on far images
FAR_EPOCHS = opt.far_epochs


########## Use current model state to train on close imgs ###########

# delete contents of checkpoints before training
# TODO add file existence check
shutil.rmtree('checkpoints/')
shutil.rmtree('close_checkpoints/')
shutil.rmtree('medium_checkpoints/')
os.mkdir('checkpoints/')

# Path to .txt files for close category
CLOSE_IMG_DIR = 'data/gate/label_categories/close/'

# get list of all img numbers in close category and shuffle
close_imgs = [img.split('.')[0] for img in os.listdir(CLOSE_IMG_DIR)]
random.shuffle(close_imgs)

#write training and valid images into txt files
print(f'number of close training images: {len(close_imgs)}, number of close valid images: {len(close_imgs)}')
print('Writing train.txt and valid.txt')
with open('data/gate/train.txt','w') as f:
    for img in close_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

with open('data/gate/valid.txt','w') as f:
    for img in close_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

#train model
subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
'--data_config','config/gate.data','--epochs', f'{CLOSE_EPOCHS}','--batch_size','4'])


# move checkpoints for close data to own directory
shutil.move('checkpoints/', 'close_checkpoints/')








########## Use current model state to train on medium imgs ###########

# Path to .txt files for medium category
MEDIUM_IMG_DIR = 'data/gate/label_categories/medium/'

# get list of all img numbers in medium category and shuffle
medium_imgs = [img.split('.')[0] for img in os.listdir(MEDIUM_IMG_DIR)]
random.shuffle(medium_imgs)

#write training and valid images into txt files
print(f'number of medium training images: {len(medium_imgs)}, number of medium valid images: {len(medium_imgs)}')
print('Writing train.txt and valid.txt')
with open('data/gate/train.txt','w') as f:
    for img in medium_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

with open('data/gate/valid.txt','w') as f:
    for img in medium_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

#train model
subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
'--data_config','config/gate.data','--epochs', f'{MEDIUM_EPOCHS}','--batch_size','4', '--pretrained_weights', f'close_checkpoints/yolov3_ckpt_{CLOSE_EPOCHS-1}.pth'])

# move checkpoints for medium data to own directory
shutil.move('checkpoints/', 'medium_checkpoints/')








########## Use current model state to train on far imgs ###########

# Path to .txt files for far category
FAR_IMG_DIR = 'data/gate/label_categories/far/'

# get list of all img numbers in far category and shuffle
far_imgs = [img.split('.')[0] for img in os.listdir(FAR_IMG_DIR)]
random.shuffle(far_imgs)

#write training and valid images into txt files
print(f'number of far training images: {len(far_imgs)}, number of far valid images: {len(far_imgs)}')
print('Writing train.txt and valid.txt')
with open('data/gate/train.txt','w') as f:
    for img in far_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

with open('data/gate/valid.txt','w') as f:
    for img in far_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

#train model
subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
'--data_config','config/gate.data','--epochs', f'{FAR_EPOCHS}','--batch_size','4', '--pretrained_weights', f'medium_checkpoints/yolov3_ckpt_{MEDIUM_EPOCHS-1}.pth'])







# ################ TEST THE MODEL #####################

for category in ['close_checkpoints/', 'medium_checkpoints/', 'checkpoints/' ]:
    # test model for each epoch, record highest mAP and write to file. 
    metrics = []

    # get each model in this category
    for model in os.listdir(category):
        weights_path = category + model

        subprocess.run(['python3', 'test.py', '--batch_size', '4', '--model_def', 'config/yolov3-gate.cfg', '--data_config', 
        'config/new_gate.data', '--weights_path', weights_path, '--class_path', 'data/gate/classes.names'])

        # save mAP score
        with open('benchmark_map.txt','r+') as f:
            mAP = f.read()
        metrics.append(mAP)

    # take the max metric mAP for this category
    max_mAP = max(metrics)
    results = open('benchmark_results.txt', 'a+')
    results.write(category + ' : ' + str(max_mAP) + '\n')
    results.close()
