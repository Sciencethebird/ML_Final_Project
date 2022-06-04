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
from networks.default_unet import default_unet

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment information
    parser.add_argument("--wandb", action='store_true', help="use wandb to log data")
    parser.add_argument("--wandb_project_name", default="ILI_ML_Final", type=str, help="your wandb project name")
    parser.add_argument("--model_name", default="default_unet", type=str, help="your segmentation model name")
    parser.add_argument("--note", default="experiment note", type=str, help="your model name")
    

    # general training settings
    parser.add_argument("--cuda_device", default="0", type=str, help="set visible cuda device")
    parser.add_argument("--epochs", default=10, type=int, help="training iteration")
    parser.add_argument("--batch_size", default=25, type=int, help="batch size")

    # deeplabv3 conficuration
    parser.add_argument("--backbone", default="ResNet50", type=str, help="encoder backbone of deeplabv3")
    # default unet configuration
    parser.add_argument("--depth", default=4, type=int, help="unet model depth")
    parser.add_argument("--ch", default=15, type=int, help="unet channel number")
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_path='datasets/ICME2022_Training_Dataset/images_real_world'#1080/1920
    label_path='datasets/ICME2022_Training_Dataset/labels_real_world'
    dataset_real = DataLoaderSegmentation(input_path,label_path,'',transforms.Resize(size=(1080,1920)))
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=args.batch_size, shuffle=True)


    inputs = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    y_ = tf.placeholder(tf.float32, [None, None, None, 6])
    x = tf.image.resize_images(inputs, (256, 256))
    x = x/255.0
    y = tf.image.resize_images(y_, (256, 256))
    b=tf.Variable(0.0)

    if args.model_name == "default_unet":
      out = default_unet(x, b, depth=args.depth, ch=args.ch)
      args.backbone = 'CNN'
    elif args.model_name =="deeplabv3+":
      out = DeeplabV3Plus(x, num_classes=6, backbone=args.backbone)

    outputs = tf.image.resize_images(out, (720, 1280))
    outputs = tf.argmax(outputs,-1)

    y_label = tf.argmax(y_,-1)
    iou, conf_mat = tf.metrics.mean_iou(labels=y_label, predictions=outputs, num_classes=6)

    loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y)
    loss=tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001)
    train = optimizer.minimize(loss+0.0005*b)
    saver=tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #saver.restore(sess, './models/')

    def stats_graph(graph):
      flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
      params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
      print('FLOPs: {}; Trainable params:{}'.format(flops.total_float_ops, params.total_parameters))

    stats_graph(tf.get_default_graph())
    #wandb.tensorflow.log(tf.summary.merge_all())
    #print(tf.summary.merge_all())
    input("press enter to continue...")


    if args.wandb == True:
      wandb.init(project=args.wandb_project_name, config=args)
      wandb.run.name = experiment_name
      wandb.run.save()
      wandb.config.update(args)

    num_epochs = args.epochs
    best_mIOU = 0.0
    for epoch in range(num_epochs):
      for i, data in enumerate(dataloader, 0):
        input = data[0].numpy()
        label = data[1].numpy()
        sess.run(train, feed_dict={inputs: input, y_: label})
        #model(input)
        #miou = sess.run(mIOU, feed_dict={inputs: input, y_: label})
        #ousst = sess.run(out, feed_dict={inputs: input, y_: label})
        #print(f"mIOU: {ousst}")
        
        #print(f"y_label: {sess.run(y_label, feed_dict={inputs: input, y_: label}).shape }")
        #print(f"out: {sess.run(out, feed_dict={inputs: input, y_: label}).shape}")
        #print(f"outputs: {sess.run(pred, feed_dict={inputs: input, y_: label}).shape}")
        #input()
        if i % 10 == 0:
          # mIOU
          sess.run(tf.local_variables_initializer()) #https://blog.csdn.net/u013841196/article/details/109533542
          sess.run([conf_mat], feed_dict={inputs: input, y_: label}) # I don't know why you need to run conf_mat first
          mIOU = sess.run([iou], feed_dict={inputs: input, y_: label})
          print(f"mIOU: {mIOU}")
          # loss
          loss_value = sess.run(loss,feed_dict={inputs: input, y_: label})
          print("[%d/%d][%s/%d] loss: %.4f b: %.4f "\
              %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), loss_value, sess.run(b)) )
          if args.wandb == True:
            wandb.log({"loss": loss_value})
            wandb.log({"mIOU": mIOU[0]})
          if mIOU[0] > best_mIOU:
            best_mIOU = mIOU
            model_path = os.path.join('./models/', experiment_name, f"ckpt_{epoch:06}", "model/")
            if not os.path.exists(model_path):
              os.makedirs(model_path)
            saver.save(sess, model_path)
            print('checkpoint saved')
          

        if i % 300==0:

          ts = time.time()
          print('checkpoint saved')
          for i, data in enumerate(dataloader_real, 0):
            input = data[0].numpy()
            label = data[1].numpy()
            sess.run(train,feed_dict={inputs: input, y_: label})
          inference_time = time.time() - ts

          if args.wandb == True:
            wandb.log({"inference time": inference_time})

          model_path = os.path.join('./models/', experiment_name, f"ckpt_{epoch:06}", "model/")
          if not os.path.exists(model_path):
              os.makedirs(model_path)
          saver.save(sess, model_path)
          print('checkpoint saved')