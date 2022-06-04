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