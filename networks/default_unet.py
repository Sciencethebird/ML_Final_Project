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

from tensorflow import keras
from tensorflow.keras import layers

def default_unet(x, b, depth=5, ch=15):
    xn = []
    x=tf.layers.conv2d(x,ch,3,1,'same')
    #x=layers.Conv2D(ch,3,1,'same')(x)
    
    x=tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    for i in range(depth):
      xn.append(x)
      x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
      if i <depth-1:
        x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
    for i in range(depth):
      if i>0:
        x = tf.keras.layers.UpSampling2D((2,2))(x)
      x = tf.layers.conv2d(x,ch*(2**(depth-i-1)),3,1,'same')+xn[-i-1]
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
    out = tf.layers.conv2d(x,6,3,1,'same')
    return out


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.layers.conv2d(
        inputs=block_input,
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
    )#(block_input)
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output
    
def deeplab_unet(x, b, depth=5, ch=15):
    xn = []
    x=tf.layers.conv2d(x,ch,3,1,'same')
    #x=layers.Conv2D(ch,3,1,'same')(x)
    
    x=tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    for i in range(depth):
      xn.append(x)
      x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
      if i <depth-1:
        x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
    
    x = DilatedSpatialPyramidPooling(x)
    for i in range(depth):
      if i>0:
        x = tf.keras.layers.UpSampling2D((2,2))(x)
      x = tf.layers.conv2d(x,ch*(2**(depth-i-1)),3,1,'same')+xn[-i-1]
      x = tf.layers.batch_normalization(x,center=False,scale=False)+b
      x = tf.nn.relu(x)
    out = tf.layers.conv2d(x,6,3,1,'same')
    return out