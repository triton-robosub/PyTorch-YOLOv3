import os
import random
import subprocess 
import shutil 


########## Use current model state to train on close imgs ###########

#delete contents of checkpoints before training
shutil.rmtree('checkpoints/')
os.mkdir('checkpoints/')

# Path to .txt files for close category
CLOSE_IMG_DIR = 'data/gate/label_categories/close/'
# Number of epochs to train model on close images
CLOSE_EPOCHS = 2

# get list of all img numbers in close category and shuffle
close_imgs = [img.split('.')[0] for img in os.listdir(CLOSE_IMG_DIR)]
random.shuffle(close_imgs)

#write training and valid images into txt files
print(f'number of training images: {len(close_imgs)}, number of valid images: {len(close_imgs)}')
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




########## Use current model state to train on medium imgs ###########

# Path to .txt files for medium category
MEDIUM_IMG_DIR = 'data/gate/label_categories/medium/'
# Number of epochs to train model on medium images
MEDIUM_EPOCHS = 2

# get list of all img numbers in medium category and shuffle
medium_imgs = [img.split('.')[0] for img in os.listdir(MEDIUM_IMG_DIR)]
random.shuffle(medium_imgs)

#write training and valid images into txt files
print(f'number of training images: {len(medium_imgs)}, number of valid images: {len(medium_imgs)}')
print('Writing train.txt and valid.txt')
with open('data/gate/train.txt','w') as f:
    for img in medium_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

with open('data/gate/valid.txt','w') as f:
    for img in medium_imgs:
        f.write('data/gate/images/' + img + ".jpg\n")

#train model
subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
'--data_config','config/gate.data','--epochs', f'{MEDIUM_EPOCHS}','--batch_size','4', '--pretrained_weights', f'yolov3_ckpt_{CLOSE_EPOCHS-1}'])





# ########## Use current model state to train on far imgs ###########

# # Path to .txt files for far category
# FAR_IMG_DIR = 'data/gate/label_categories/far/'
# # Number of epochs to train model on far images
# FAR_EPOCHS = 10

# # get list of all img numbers in far category and shuffle
# far_imgs = [img.split('.')[0] for img in os.listdir(FAR_IMG_DIR)]
# random.shuffle(far_imgs)

# #write training and valid images into txt files
# print(f'number of training images: {len(far_imgs)}, number of valid images: {len(far_imgs)}')
# print('Writing train.txt and valid.txt')
# with open('data/gate/train.txt','w') as f:
#     for img in far_imgs:
#         f.write('data/gate/images/' + img + ".jpg\n")

# with open('data/gate/valid.txt','w') as f:
#     for img in far_imgs:
#         f.write('data/gate/images/' + img + ".jpg\n")

# #train model
# subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
# '--data_config','config/gate.data','--epochs', f'{FAR_EPOCHS}','--batch_size','4', '--pretrained_weights', f'yolov3_ckpt_{MEDIUM_EPOCHS-1}'])




# ################ TEST THE MODEL #####################

# #test model for each epoch, record highest mAP and write to file. 
# metrics = []

# for i in range(0, EPOCHS + 1, 5):
#     if i == EPOCHS:
#         i = EPOCHS - 1
#     weights_path = f'checkpoints/yolov3_ckpt_{i}.pth'

#     subprocess.run(['python3', 'test.py', '--batch_size', '4', '--model_def', 'config/yolov3-gate.cfg', '--data_config', 
#     'config/new_gate.data', '--weights_path', weights_path, '--class_path', 'data/gate/classes.names'])

#     with open('benchmark_map.txt','r+') as f:
#         mAP = f.read()
#     metrics.append(mAP)

# max_mAP = max(metrics)
# results = open('benchmark_results.txt', 'a+')
# results.write(str(num_imgs) + 'images :' + str(max_mAP) + '\n')
# results.close()
