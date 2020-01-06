# Training
```
python3 -W ignore train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 100 --batch_size 4 --pretrained_weights weights/yolov3.weights
```

# Detecting
```
python3 detect.py --image_folder data/custom/images/ --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names --checkpoint_model checkpoints/yolov3_ckpt_99.pth
```