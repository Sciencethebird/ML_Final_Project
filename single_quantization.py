import cv2
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tensorflow as tf
import argparse

from pathlib import Path
print("Tensorflow Version is %s" % tf.__version__)

#Preparing your calibration data
def representative_dataset():
  glob =""
  glob += args.glob
  glob += ".jpg"
  print("glob =",glob)
  for data in Path('./images_real_world/').glob(glob):
  # for data in Path('./Testing_Data_for_Qualification/').glob(glob):
    img = cv2.imread(str(data))
    img = np.expand_dims(img,0)
    img = img.astype(np.float32)
    yield [img]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # experiment information
  parser.add_argument("--output_name", default="quantized_model.tflite", type=str, help=".tflite name")
  parser.add_argument("--glob", default="*.jpg", type=str, help="glob")
  args = parser.parse_args()
  #Preparing your FP32 model (Please refer to Lab#1) and do post training quantization by NN API (https://www.tensorflow.org/lite/performance/post_training_quantization)
  converter = tf.lite.TFLiteConverter.from_frozen_graph(
      graph_def_file = './Final_model.pb',
      input_arrays = ['Placeholder'],
      input_shapes = {'Placeholder':[1, 1080, 1920,3]},
      output_arrays = ['ArgMax'],
  )
  # converter.optimizations = []
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  tflite_model = converter.convert()
  open('./tflite/'+args.output_name, 'wb').write(tflite_model)
  # open('./tflite/'+args.output_name+"_test", 'wb').write(tflite_model)
