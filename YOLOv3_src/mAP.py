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
# modelWeightsPath = "/Data/models/22.05.30/rgb_yolo"
modelWeightsPath = "/Data/models/dec02/yolov3"

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

def postprocess_gtboxes(gtbox, orig_img_shape, input_size):
    coordinates = np.concatenate([gtbox[:,:2] - 0.5*gtbox[:,2:],
                                  gtbox[:,:2] + 0.5*gtbox[:,2:]], axis=-1)
    org_h, org_w = orig_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    coordinates[:, 0::2] = 1.0 * (coordinates[:, 0::2] - dw) / resize_ratio
    coordinates[:, 1::2] = 1.0 * (coordinates[:, 1::2] - dh) / resize_ratio

    return coordinates

def bboxes_AP(preds, t_corners):
    p_corners = np.array(preds)[:,:4] #x, y, x2, y2, conf, pred
    #targets: x, y, x2, y2

    p_count = len(p_corners)
    t_count = len(t_corners)

    p_wh = p_corners[...,2:] - p_corners[...,:2]
    t_wh = t_corners[...,2:] - t_corners[...,:2]

    p_area = np.product(p_wh, axis=-1)
    t_area = np.product(t_wh, axis=-1)

    p_area = np.stack([p_area]*t_count, axis=-1)
    t_area = np.stack([t_area]*p_count, axis=0)

    left  = np.stack([np.maximum(p_corners[:,0], ti) for ti in t_corners[:,0]], axis=-1)
    up    = np.stack([np.maximum(p_corners[:,1], ti) for ti in t_corners[:,1]], axis=-1)
    right = np.stack([np.minimum(p_corners[:,2], ti) for ti in t_corners[:,2]], axis=-1)
    down  = np.stack([np.minimum(p_corners[:,3], ti) for ti in t_corners[:,3]], axis=-1)

    i_w = np.maximum(right - left, 0.0)
    i_h = np.maximum(down - up, 0.0)
    i_area = i_h * i_w

    union = p_area + t_area - i_area

    IoU = 1.0 * i_area / np.maximum(union, 1e-12)

    """Method 1: All preds get best IoU available. Not all targets get preds"""
    IoUs = np.amax(IoU, axis=1)
    return list(IoUs >= 0.5)

    """Method 2: Each target gets only max one pred. Remaining preds have no target"""
    # diff = t_count - p_count
    # IoU = np.pad(IoU, ((0, max(diff, 0)), (0, max(-diff, 0))))

    # ious = [0]*max(p_count, t_count)

    # for i in range(len(IoU)):
    #     colmax = np.amax(IoU, axis=0)
    #     col = np.argmax(colmax)
    #     row = np.argmax(IoU[:,col])
    #     ious[row] = IoU[row,col]
    #     IoU[:,col] = -1
    #     IoU[row,:] = -1
        
    # precision = np.round(ious)
    # precision[precision==-1] = 0

    # return list(precision)

    """n-n mapping of targets and preds. Method 1 evaluated over many thresholds""" 
    #
    # IoUs = np.amax(IoU, axis=1)
    # diff = t_count - p_count
    # squareIOU = np.pad(IoU, ((0, max(diff, 0)), (0, max(-diff, 0))))

    # thresholds = np.arange(0,1.01,0.01)
    # ps = []
    # rs = []
    # for thresh in thresholds:
    #     tp = np.sum(np.amax(IoU >= thresh, axis=-1))
    #     fp = np.sum(np.amin(IoU < thresh, axis=-1))
    #     fn = np.sum(np.amax(IoU < thresh, axis=0))
        
    #     ps.append(tp/(tp+fp))
    #     rs.append(tp/(tp+fn))
        

    # rs.append(0)
    # ps.append(1)

    # rs = np.array(rs)
    # ps = np.array(ps)
    
    # ap = np.sum((rs[:-1] - rs[1:]) * ps[:-1])
    # return [ap]

        # Needs to now find mean value




def test_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_bbox = model(image_data)
        
        precision = []

        for i in range(cfg.TEST_BATCH_SIZE):
            pbox = [pred_bbox[1][i,:], pred_bbox[3][i,:], pred_bbox[5][i,:]]
            pbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pbox]
            pbox = tf.concat(pbox, axis=0)
            box = utils.postprocess_boxes(pbox, (720,1280), input_size[0], 0.3)
            box = utils.nms(box, 0.25, method='soft-nms')
            box = np.array(box)
            # bboxes.append(box)

            tbox = [target[0][1][i,:], target[1][1][i,:], target[2][1][i,:]]
            tbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in tbox]
            tbox = tf.concat(tbox, axis=0)
            gtbox = tbox.numpy()
            gtbox = gtbox[~np.all(tbox==0, axis=1)]
            gtbox = postprocess_gtboxes(gtbox, (720,1280), input_size[0])

            p_count = len(box)
            t_count = len(gtbox)

            if p_count == 0:
                precision += [0] * t_count
            elif t_count == 0:
                precision += [0] * p_count
            else:
                precision += bboxes_AP(box, gtbox)

        if precision:
            precision = np.mean(precision)
        else:
            precision = -1

        tf.print("=> STEP %4d   AP: %4.2f" %(global_steps, precision))
                                                          
        global_steps.assign_add(1)
        return precision


# total_giou = 0
total_AP = []
counter = 0

for image_data, target in testset:
    AP = test_step(image_data, target)
    counter += cfg.TEST_BATCH_SIZE
    # total_giou += giou
    if AP == -1:
        continue
    else:
        total_AP.append(AP)


# tf.print("Total GIoU: %4.2f\tImages in testset: %4d\tAverage GIoU: %4.2f" %(total_giou, counter, total_giou/counter))
# print("Total GIoU: ",total_giou,"\t\tImages iterated over: ",total_giou/counter)
print("Average IoU: ",np.mean(total_AP),"\t\tImages iterated over: ",counter)


print("Finished trawling test set")
