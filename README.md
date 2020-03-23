# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Pre-Installation for Root User
1. Create user and give sudo permissions.
```
sudo adduser [username]
usermod -aG sudo [username]
```
2. Copy over robosub conda environment requirements file to the new user.
```
sudo cp -r ~/robosub/yolov3/robosub_conda_env.txt /home/[username]
```

## Pre-Installation for Regular User
- Link to [original repository](https://github.com/eriklindernoren/PyTorch-YOLOv3.git).
- Link to [Triton Robosub repository](https://github.com/triton-robosub/PyTorch-YOLOv3).
1. Use TMUX always. Refer to this [repo](https://gist.github.com/MohamedAlaa/2961058) for TMUX cheatsheet.
```
tmux a
```
2. Install Anaconda on Ubuntu 18.04.
```
sh install_anaconda.sh && exit
```
3. Create Robosub conda environment.
```
conda create --name robosub --file robosub_conda_env.txt
```
4. Activate the robosub conda environment.
```
conda activate robosub
```
5. Clone this repository.
```
git clone https://github.com/triton-robosub/PyTorch-YOLOv3.git
```
6. Confirm that nobody else is using the GPU. Under "Volatile GPU-Util" the pecentage should be 0% if it is not being used. If it is being used, it will be approximately 80% or above.
```
watch nvidia-smi
```
<p align="center"><img src="assets/GPU-output.png" width="480"\></p>

## YOLOV3 Training
### Training parameters
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

### Training with pretrained weights
```
python3 -W ignore train.py --model_def config/yolov3-gate.cfg --data_config config/gate.data --epochs 100 --batch_size 4 --pretrained_weights weights/yolov3.weights
```

### Training with new weights
```
python3 -W ignore train.py --model_def config/yolov3-gate.cfg --data_config config/gate.data --epochs 100 --batch_size 4
```

### Detecting
```
python3 detect.py --image_folder /home/noah/robosub/data/gate/new_gate/images --model_def config/yolov3-gate.cfg --class_path data/gate/classes.names --checkpoint_model /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth --batch_size 4
```

### Testing
```
python3 test.py --batch_size 4 --model_def config/yolov3-gate.cfg --data_config config/new_gate.data --weights_path /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth --class_path data/gate/classes.names 
```

### Live
```
python3 live.py --img_path data/gate/images/177.jpg --checkpoint_model /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth
```