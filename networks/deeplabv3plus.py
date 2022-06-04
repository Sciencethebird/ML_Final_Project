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

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
    )(block_input)
    x = layers.BatchNormalization()(x)
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

#def DeeplabV3Plus(image_size, num_classes):
def DeeplabV3Plus(model_input, image_size=256, num_classes=6, backbone="ResNet50"):
    #model_input = keras.Input(shape=(image_size, image_size, 3))
    if backbone == "ResNet50":
        encoder = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
    elif backbone == "MobileNet":
        encoder = keras.applications.MobileNet(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
    elif backbone == "MobileNetV2":
        encoder = keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
    else:
        print("no such network")
    
    print(encoder.summary())
    input("press enter to continue...")

    if backbone == "ResNet50":
        x = encoder.get_layer("conv4_block6_2_relu").output
    elif backbone == "MobileNet":
        x = encoder.get_layer("conv_pw_12_relu").output
    elif backbone == "MobileNetV2":
        x = encoder.get_layer("block_15_add").output
    else:
        print("no such network")
    
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    if backbone == "ResNet50":
        input_b = encoder.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    elif backbone == "MobileNet":
        input_b = encoder.get_layer("conv_pw_3_relu").output
        input_b = convolution_block(input_b, num_filters=256, kernel_size=1)
    elif backbone == "MobileNetV2":
        input_b = encoder.get_layer("block_2_add").output
        input_b = convolution_block(input_b, num_filters=256, kernel_size=1)
    else:
        print("no such network")

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return model_output