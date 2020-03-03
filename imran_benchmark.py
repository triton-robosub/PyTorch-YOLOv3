import os
import random
import subprocess 
import shutil 

CLOSE_IMG_DIR = 'data/gate/label_categories/close/'
MEDIUM_IMG_DIR = 'data/gate/label_categories/medium/'
FAR_IMG_DIR = 'data/gate/label_categories/far/'
EPOCHS = 50

# get the image numbers based on group and shuffle within group
close_imgs = [img.split('.')[0] for img in os.listdir(CLOSE_IMG_DIR)]
medium_imgs = [img.split('.')[0] for img in os.listdir(MEDIUM_IMG_DIR)]
far_imgs = [img.split('.')[0] for img in os.listdir(FAR_IMG_DIR)]
random.shuffle(close_imgs)
random.shuffle(medium_imgs)
random.shuffle(far_imgs)

#get random subset of images of size num_imgs, split 80/20 train/valid
train_size = int(num_imgs * 0.8)
train_img = cur_subset[:train_size]
valid_img = cur_subset[train_size:num_imgs]

#write training and valid images into txt files
print(f'number of training images: {len(train_img)}, number of valid images: {len(valid_img)}')
print('Writing train.txt and valid.txt')
with open('data/gate/train.txt','w') as f:
    for img in train_img:
        f.write('data/gate/images/' + img + ".jpg\n")

with open('data/gate/valid.txt','w') as f:
    for img in valid_img:
        f.write('data/gate/images/' + img + ".jpg\n")

#delete contents of checkpoints before training
shutil.rmtree('checkpoints/')
os.mkdir('checkpoints/')


# #train model
# subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
# '--data_config','config/gate.data','--epochs', f'{EPOCHS}','--batch_size','4'])

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
