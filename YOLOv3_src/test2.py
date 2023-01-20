import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dataset import Dataset
from yolov3 import YOLOv3, decode, compute_loss
import config as cfg
import utils

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
    output_tensors.append(conv_tensor) # Troublesome for the 
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
model.load_weights(modelWeightsPath)
optimizer = tf.keras.optimizers.Adam()
metrics = [tf.keras.metrics.MeanIoU(1),
           tf.keras.metrics.BinaryAccuracy()]
callbacks = tf.keras.callbacks.TensorBoard(log_dir=logdir+'profile', histogram_freq=1,
                                           profile_batch=2)
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

def get_iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
        
    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box
    
    y11, x11, y21, x21, _ = box1
    y12, x12, y22, x22, _ = box2
    
    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou


def test_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model.predict(image_data, callbacks = [callbacks])
        giou_loss=conf_loss=prob_loss=iou=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
            # iou += loss_items[3]

        # iou = get_iou(box1, box2)

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f   time: %s  " %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss, time.asctime()))
        # # update learning rate
        global_steps.assign_add(1)

        # # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


total_giou = 0
total_iou = []
counter = 0

for image_data, target in testset:
    test_step(image_data, target)
    counter += cfg.TEST_BATCH_SIZE
    # total_giou += giou
    # total_iou.append(iou)


tf.print("Total GIoU: %4.2f\tImages in testset: %4d\tAverage GIoU: %4.2f" %(total_giou, counter, total_giou/counter))
# print("Total GIoU: ",total_giou,"\t\tImages iterated over: ",total_giou/counter)


print("Finished trawling test set")
