import os
import random
import subprocess 
import shutil 

# upperbound for number of total training labels, currently 1812
MAX_IMGS = 1900 
# anything less will result in errors during validation stage.
START_IMGS = 700 
# TODO
STEP_SIZE = 100

# Read in path to data files
print("What is the path to the data folder that contains the images, labels, etc.? For example data/custom.")
DATA_DIR = input("Path: ") 

print("What is the number of epochs for this benchmark?")
EPOCHS = int(input("Epochs: "))

all_imgs = os.listdir(os.path.join(DATA_DIR, "labels"))
all_imgs = [img.split('.')[0] for img in all_imgs]
random.shuffle(all_imgs)

for num_imgs in range(START_IMGS, MAX_IMGS, STEP_SIZE):
    print(num_imgs)

    #always want to include same images as previous num_imgs
    #ex. 800 images has same 700 + 100 additional random images, for consistency
    cur_subset = all_imgs[:num_imgs]
    random.shuffle(cur_subset)
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

    #train model
    subprocess.run(['python3','-W','ignore','train.py','--model_def','config/yolov3-gate.cfg',
    '--data_config','config/gate.data','--epochs', f'{EPOCHS}','--batch_size','4'])

    #test model for each epoch, record highest mAP and write to file. 
    metrics = []

    for i in range(0, EPOCHS + 1, 1):
        if i == EPOCHS:
            i = EPOCHS - 1
        weights_path = f'checkpoints/yolov3_ckpt_{i}.pth'

        subprocess.run(['python3', 'test.py', '--batch_size', '4', '--model_def', 'config/yolov3-gate.cfg', '--data_config', 'config/gate.data', '--weights_path', weights_path, '--class_path', 'data/gate/classes.names'])

        with open('benchmark_map.txt','r+') as f:
            mAP = f.read()
        metrics.append(mAP)

    max_mAP = max(metrics)
    results = open('benchmark_results.txt', 'a+')
    results.write(str(num_imgs) + 'images :' + str(max_mAP) + '\n')
    results.close()
