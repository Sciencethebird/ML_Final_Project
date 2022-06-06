from email.policy import default
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tensorflow as tf
import time

import wandb
import argparse
from pathlib import Path
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from datasets.dataloader import DataLoaderSegmentation

from networks.deeplabv3plus import DeeplabV3Plus
from networks.default_unet import deeplab_unet, default_unet

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment information
    parser.add_argument("--wandb", action='store_true', help="use wandb to log data")
    parser.add_argument("--wandb_project_name", default="ILI_ML_Final", type=str, help="your wandb project name")
    parser.add_argument("--model_name", default="default_unet", type=str, help="your segmentation model name")
    parser.add_argument("--note", default="experiment note", type=str, help="your model name")
    
    # general training settings
    parser.add_argument("--cuda_device", default="0", type=str, help="set visible cuda device")
    parser.add_argument("--epochs", default=40, type=int, help="training iteration")
    parser.add_argument("--batch_size", default=25, type=int, help="batch size")
    parser.add_argument("--lr_init", default=0.001, type=float, help="start learning rate value")
    parser.add_argument("--lr_decay", default=0.85, type=float, help="decay coeff for exponentail decay schedular")
    parser.add_argument("--lr_decay_step_rate", default=4000, type=int, help="step to use for the decay computation")

    # deeplabv3 conficuration
    parser.add_argument("--backbone", default="ResNet50", type=str, help="encoder backbone of deeplabv3")

    # default_unet and deeplab_unet configuration
    parser.add_argument("--depth", default=4, type=int, help="unet model depth")
    parser.add_argument("--ch", default=15, type=int, help="unet channel number")
    
    # checkpoint operation
    parser.add_argument("--ckpt", default=None, type=str, help="source checkpoint")
    parser.add_argument("--pb", default="test.pb", type=str, help="output pb file name")
    args = parser.parse_args()
    return args

print("Tensorflow Version is %s" % tf.__version__)

if __name__ == '__main__':

    args = parse_arguments()
    experiment_name = f"{args.model_name}:{datetime.today().strftime('%Y-%m-%d-%H-%M')}"

    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

    input_path='datasets/ICME2022_Training_Dataset/images'#720/1280
    label_path='datasets/ICME2022_Training_Dataset/labels/class_labels'
    dataset = DataLoaderSegmentation(input_path,label_path,'_lane_line_label_id',transforms.Resize(size=(720,1280)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    input_path='datasets/ICME2022_Training_Dataset/images_real_world'#1080/1920
    label_path='datasets/ICME2022_Training_Dataset/labels_real_world'
    dataset_real = DataLoaderSegmentation(input_path,label_path,'',transforms.Resize(size=(1080,1920)))
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=1, shuffle=True)


    inputs = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    y_ = tf.placeholder(tf.float32, [None, None, None, 6])
    x = tf.image.resize_images(inputs, (256, 256))
    x = x/255.0
    y = tf.image.resize_images(y_, (256, 256))
    b=tf.Variable(0.0)

    if args.model_name == "default_unet":
      out = default_unet(x, b, depth=args.depth, ch=args.ch)
      args.backbone = 'CNN'
    elif args.model_name == "deeplab_unet":
      out = deeplab_unet(x, b, depth=args.depth, ch=args.ch)
      args.backbone = 'CNN'
    elif args.model_name =="deeplabv3+":
      out = DeeplabV3Plus(x, num_classes=6, backbone=args.backbone)

    outputs = tf.image.resize_images(out, (1080, 1920))
    outputs = tf.argmax(outputs, -1)

    loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y)
    loss=tf.reduce_mean(loss)

    # learning rate
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(args.lr_init, global_step, args.lr_decay_step_rate, args.lr_decay, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) #0.00001)
    #train = optimizer.minimize(loss+0.0005*b)
    train = optimizer.minimize(loss)
    saver=tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, f"./models/{args.ckpt}/model/")

    def stats_graph(graph):
      flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
      params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
      print('FLOPs: {}; Trainable params:{}'.format(flops.total_float_ops, params.total_parameters))

    stats_graph(tf.get_default_graph())
    #wandb.tensorflow.log(tf.summary.merge_all())
    #print(tf.summary.merge_all())
    #input("press enter to continue...")

    graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names = ['ArgMax'])
    tf.train.write_graph(graph_def, 'pb', args.pb, as_text = False)
