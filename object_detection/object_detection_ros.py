#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME= 'ssd_inception_v2_coco_11_06_2017'
# MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = sys.path[0] + '/../models' + '/' + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(sys.path[0], 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Mamually Install
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# Load a Tensorflow mode into memory
def load_frozenmodel():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_labelmap():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

class ObjectDetection(object):
    def __init__(self, detection_graph, category_index):
        self.image_sub = rospy.Subscriber("/image", Image, self.imageCallback, queue_size=10)
        self.image_pub = rospy.Publisher("/object_detection/image", Image, queue_size=10)
        self.detection_graph = detection_graph
        self.category_index  = category_index
        self.flag = False

    def imageCallback(self, image_msg):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
            self.flag = True
        except CvBridgeError as e:
            print (e)

    def setConfig(self, per, dev):
        self.config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=per,
                    visible_device_list=dev,
                    allow_growth=True
                    )
                )
    def object_detection(self, sess):
        image_np = self.cv_image
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor      = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes             = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores            = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes           = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections    = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

        image_height = self.cv_image.shape[0]
        image_width  = self.cv_image.shape[1]
        resize_image = cv2.resize(image_np, (image_width, image_height))
        pub_image = CvBridge().cv2_to_imgmsg(resize_image, "bgr8")
        self.image_pub.publish(pub_image)

    def main(self):
        rospy.init_node("object_detection_ros")
        rate = rospy.Rate(30)
        per = rospy.get_param('~per', 0.4)
        dev = rospy.get_param('~dev', "0")
        # setup config
        self.setConfig(per, dev)
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=self.config) as sess:
                while not rospy.is_shutdown():
                    if self.flag:
                        self.object_detection(sess)
                        self.flag = False
                    rate.sleep()

def main():
    # Load 
    category = load_labelmap()
    graph = load_frozenmodel()
    # Detection
    detection = ObjectDetection(graph, category)
    detection.main()

if __name__ == '__main__':
    print("start")
    main()
