import os
import shutil
import tensorflow as tf
import numpy as np
from time import time

from dataset import Dataset
from yolov3 import YOLOv3, decode
import config as cfg


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
modelWeightsPath = "/Data/models/22.05.30/rgb_yolo"
# modelWeightsPath = "/Data/models/dec02/yolov3"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

testset = Dataset('test')
logdir = "./log"
steps_per_epoch = len(testset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

input_size = (416,416)
input_tensor = tf.keras.layers.Input([*input_size, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
model.load_weights(modelWeightsPath)
optimizer = tf.keras.optimizers.Adam()
metrics = [tf.keras.metrics.MeanIoU(1),
           tf.keras.metrics.BinaryAccuracy()]
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

time()

times = []
for image_data, target in testset:
    t1 = time()
    model(image_data)
    t2 = time()
    times.append(t2 - t1)
    print(t2 - t1)

print(np.average(times))