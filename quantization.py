import cv2
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tensorflow as tf

from pathlib import Path
print("Tensorflow Version is %s" % tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Preparing your calibration data
def representative_dataset():
  for data in Path('datasets/Testing_Data_for_Qualification/').glob('*.jpg'):
    img = cv2.imread(str(data))
    img = np.expand_dims(img,0)
    img = img.astype(np.float32)
    yield [img]

#Preparing your FP32 model (Please refer to Lab#1) and do post training quantization by NN API (https://www.tensorflow.org/lite/performance/post_training_quantization)
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = './pb/test2.pb',
    input_arrays = ['Placeholder'],
    input_shapes = {'Placeholder':[1, 1080, 1920, 3]},
    output_arrays = ['ArgMax'],
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
open('./lab2_quantized_model.tflite', 'wb').write(tflite_model)

'''
img = cv2.imread('./Testing_Data_for_Qualification/0002.jpg')
plt.imshow(img)

''img = np.expand_dims(img,0)
img = img.astype(np.float32)

interpreter = tf.lite.Interpreter('./lab2_quantized_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

plt.imshow(output[0])'''''