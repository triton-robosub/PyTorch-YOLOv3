## Training with pretrained weights
```
python3 -W ignore train.py --model_def config/yolov3-gate.cfg --data_config config/gate.data --epochs 100 --batch_size 4 --pretrained_weights weights/yolov3.weights
```

## Training with new weights
```
python3 -W ignore train.py --model_def config/yolov3-gate.cfg --data_config config/gate.data --epochs 100 --batch_size 4
```

## Detecting
```
python3 detect.py --image_folder /home/noah/robosub/data/gate/new_gate/images --model_def config/yolov3-gate.cfg --class_path data/gate/classes.names --checkpoint_model /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth --batch_size 4
```

## Testing
```
python3 test.py --batch_size 4 --model_def config/yolov3-gate.cfg --data_config config/new_gate.data --weights_path /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth --class_path data/gate/classes.names 
```

<<<<<<< HEAD
## Live
python3 live.py --img_path data/gate/images/177.jpg --checkpoint_model /home/noah/robosub/models/yolov3/gate20/yolov3_ckpt_19.pth
=======

>>>>>>> 68606c154348192cdf2f18e5815530bb6c859a70
