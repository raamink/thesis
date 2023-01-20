import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from dataset import Dataset
from yolov3 import YOLOv3, decode, compute_loss
import config as cfg
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
modelWeightsPath = "/Data/models/22.05.30/rgb_yolo"
predictions = Path("/Data/models/22.05.30/predictions-rgb-testset.txt")
# modelWeightsPath = "/Data/models/dec02/yolov3"
# predictions = Path("/Data/models/dec02/predictions-flow-testset.txt")
if predictions.exists():
    predictions.unlink()
predictions.touch()
predFile = predictions.open('a')


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
    if ((xi2-xi1) < 0) or ((yi2-yi1) < 0): 
        # That means no intersection, as on one or both axis they do not
        # overlap. When both are True, the else would not catch it.
        inter_area = 0
    else:
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou

def bboxes_iou(preds, t_corners):
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

    #
    IoUs = np.amax(IoU, axis=1)
    return list(IoUs)

    """If only the best box gets a valid IoU, use below. Else above"""
    # diff = t_count - p_count
    # IoU = np.pad(IoU, ((0, max(diff, 0)), (0, max(-diff, 0))))

    # ious = [0]*max(p_count,t_count)

    # for i in range(len(IoU)):
    #     colmax = np.amax(IoU, axis=0)
    #     col = np.argmax(colmax)
    #     row = np.argmax(IoU[:,col])
    #     ious[row] = IoU[row,col]
    #     IoU[:,col] = -1
    #     IoU[row,:] = -1

    # return ious


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


def test_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_bbox = model(image_data)
        
        # ious = []
        # gious = []

        # for i in range(3):
        #     conv = pred_bbox[2*i]
        #     pred = pred_bbox[2*i+1]
        #     label, bboxes = target[i]

        #     label_xywh = label[:,:,:,:, 0:4]
        #     pred_xywh = pred[:,:,:,:, 0:4]

        #     label_corners = tf.concat[label_xywh[...,:2] - label_xywh[...,2:]*0.5, label_xywh[...,:2]+label_xywh[...,2:]*0.5], axis=-1)
        #     pred_corners = tf.concat([pred_xywh[...,:2] - pred_xywh[...,2:]*0.5, pred_xywh[...,:2]+pred_xywh[...,2:]*0.5], axis=-1)

        #     label_area = label_xywh[...,2]* label_xywh[...,3]
        #     pred_area = pred_xywh[...,2]*pred_xywh[...,3]

        #     left_up = tf.maximum(pred_corners[...,:2], label_corners[...,:2])
        #     right_down = tf.minimum(pred_corners[...,2:], label_corners[...,2:])

        #     i_lengths = tf.maximum(right_down-left_up, 0.0)
        #     intersect = i_lengths[...,0], i_lengths[...,1]

        #     union = label_area + pred_area - intersect
            
        #     iou = 1.0 * intersect / tf.maximum(union, 1e-12)

        #     enclose_left_up = tf.minimum(pred_corners[...,:2], label_corners[...,:2])
        #     enclose_right_down = tf.maximum(pred_corners[...,2:], label_corners[...,2:])
        #     enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        #     enclose_area = enclose[...,0] * enclose[...,1]

        #     giou = iou - 1.0 * (enclose_area - union) / tf.maximum(enclose_area, 1e-12)

        #     ious.append(iou)
        #     gious.append(giou)


        iou = []

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
                pass
            elif t_count == 0:
                iou += [0] * p_count
            else:
                iou += bboxes_iou(box, gtbox)
            # print(testset.batchPaths[i],p_count)
            strbox = ' '.join(','.join(str(j) for j in i) for i in box)
            predFile.write(f'{testset.batchPaths[i]};{strbox}\n')
            
                
            # giou.append(bbox_giou(bboxes, target[i,...]))
        
        if iou:
            iou = np.mean(iou)
        else:
            iou = 0


        # gradients = tape.gradient(total_loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   IoU: %4.2f" %(global_steps, iou))
                                                          
        global_steps.assign_add(1)
        return iou


# total_giou = 0
total_iou = []
counter = 0

for image_data, target in testset:
    iou = test_step(image_data, target)
    counter += cfg.TEST_BATCH_SIZE
    # total_giou += giou
    total_iou.append(iou)


# tf.print("Total GIoU: %4.2f\tImages in testset: %4d\tAverage GIoU: %4.2f" %(total_giou, counter, total_giou/counter))
# print("Total GIoU: ",total_giou,"\t\tImages iterated over: ",total_giou/counter)
print("Average IoU: ",np.mean(total_iou),"\t\tImages iterated over: ",counter)


print("Finished trawling test set")
