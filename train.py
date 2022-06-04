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

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment information
    parser.add_argument("--wandb", action='store_true', help="use wandb to log data")
    parser.add_argument("--wandb_project_name", default="ILI_ML_Final", type=str, help="your wandb project name")
    parser.add_argument("--model_name", default="default-Unet", type=str, help="your model name")
    parser.add_argument("--note", default="experiment note", type=str, help="your model name")
    

    # general training settings
    parser.add_argument("--cuda_device", default="0", type=str, help="set visible cuda device")
    parser.add_argument("--epochs", default=10, type=int, help="training iteration")
    parser.add_argument("--batch_size", default=25, type=int, help="batch size")
    
    # deeplabv3 conficuration
    parser.add_argument("--backbone", default="ResNet50", type=str, help="encoder backbone of deeplabv3")
    # default unet configuration
    parser.add_argument("--depth", default=4, type=int, help="unet model depth")
    args = parser.parse_args()
    return args

print("Tensorflow Version is %s" % tf.__version__)

if __name__ == '__main__':

    args = parse_arguments()
    experiment_name = f"{args.model_name}-{datetime.today().strftime('%Y-%m-%d-%H-%M')}"

    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_device

    input_path='datasets/ICME2022_Training_Dataset/images'#720/1280
    label_path='datasets/ICME2022_Training_Dataset/labels/class_labels'
    dataset = DataLoaderSegmentation(input_path,label_path,'_lane_line_label_id',transforms.Resize(size=(720,1280)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_path='datasets/ICME2022_Training_Dataset/images_real_world'#1080/1920
    label_path='datasets/ICME2022_Training_Dataset/labels_real_world'
    dataset_real = DataLoaderSegmentation(input_path,label_path,'',transforms.Resize(size=(1080,1920)))
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=args.batch_size, shuffle=True)


    #model = Deeplabv3(weights='pascal_voc', input_tensor=None, 
    #          input_shape=(256, 256, 3), classes=6, backbone='mobilenetv2',OS=16, alpha=1.)

    inputs = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    y_ = tf.placeholder(tf.float32, [None, None, None, 6])
    x = tf.image.resize_images(inputs, (256, 256))
    x = x/255.0
    y = tf.image.resize_images(y_, (256, 256))
    ch=15
    #depth=args.depth
    #xn = []
    b=tf.Variable(0.0)
    #x=tf.layers.conv2d(x,ch,3,1,'same')
    #x=tf.layers.batch_normalization(x)
    #x = tf.nn.relu(x)
    #for i in range(depth):
    #  xn.append(x)
    #  x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
    #  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    #  x = tf.nn.relu(x)
    #  x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
    #  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    #  x = tf.nn.relu(x)
    #  if i <depth-1:
    #    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
    #for i in range(depth):
    #  if i>0:
    #    x = tf.keras.layers.UpSampling2D((2,2))(x)
    #  x = tf.layers.conv2d(x,ch*(2**(depth-i-1)),3,1,'same')+xn[-i-1]
    #  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
    #  x = tf.nn.relu(x)
    #out = tf.layers.conv2d(x,6,3,1,'same')
    out = DeeplabV3Plus(x, num_classes=6, backbone=args.backbone)
    outputs = out
    outputs = tf.image.resize_images(outputs, (720, 1280))
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
          saver.save(sess, './models/')
      saver.save(sess, './models/')
      print('checkpoint saved')