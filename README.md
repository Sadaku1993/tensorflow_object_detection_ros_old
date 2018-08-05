# Tensorflow Object Detection API for ROS
In this repository, I compiled the source code using ROS and Pyhon2.7

## Overview
[tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) for ROS

## Requirements
- ROS Kinetic(ubuntu 16.04)
- Python2.7+
- [Opencv](https://opencv.org/)3.3+
- [tensorflow](https://www.tensorflow.org/install/)1.4+
- [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)

## How to Install
### Tensorflow object detection API
- [tensorflow object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
### ROS Kinetic
- [ROS Kinetic installation](http://wiki.ros.org/ja/kinetic/Installation/Ubuntu)
### Clone this Repository
```
$ git clone https://github.com/Sadaku1993/tensorflow_object_detection
$ cd catkin_ws
$ catkin make
```

## Download model
```
$ roscd tensorflow_object_detection/models
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz 
```

## How to RUN
### Change model name 
```
$ roscd tensorflow_object_detection/object_detefction
$ vim object_detection_ros.py
```
**object_detection_ros.py**
```python
    " MODEL_NAME= 'ssd_inception_v2_coco_11_06_2017'"
    " # MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'"
    " # MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'"
    " # MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'"
```

### Change param(topic name)
```
$ roscd tensorflow_object_detection/launch
$ vim object_detection_ros.launch
```

**object_detection.launch**
```
per : GPUの使用率
dev : GPU device
image : Subscribe Topic(sensor_msgs/Image)
object_detection/image : Publish Topic(sensor_msgs/Image)
```
