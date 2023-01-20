import numpy as np
import tensorflow as tf
import common as common
import config as cfg

NUM_CLASS       = len(cfg.YOLO_CLASSES)
ANCHORS         = np.array(cfg.YOLO_ANCHORS)
STRIDES         = np.array(cfg.YOLO_STRIDES)
IOU_LOSS_THRESH = cfg.YOLO_IOU_LOSS_THRESH

def darknet53(input_data):
    # 416x416x3
    input_data = common.convolutional(input_data, (3, 3,  3,  32//cfg.YOLO_DIV)) # 416x416x32
    input_data = common.convolutional(input_data, (3, 3, 32//cfg.YOLO_DIV,  64//cfg.YOLO_DIV), downsample=True) #208x208x64

    for i in range(1):
        input_data = common.residual_block(input_data,  64//cfg.YOLO_DIV,  32//cfg.YOLO_DIV, 64//cfg.YOLO_DIV) # 208x208x64

    input_data = common.convolutional(input_data, (3, 3,  64//cfg.YOLO_DIV, 128//cfg.YOLO_DIV), downsample=True) # 104x104x128

    for i in range(1):
        input_data = common.residual_block(input_data, 128//cfg.YOLO_DIV,  64//cfg.YOLO_DIV, 128//cfg.YOLO_DIV) # 104x104x128

    input_data = common.convolutional(input_data, (3, 3, 128//cfg.YOLO_DIV, 256//cfg.YOLO_DIV), downsample=True) # 52x52x256

    for i in range(1):
        input_data = common.residual_block(input_data, 256//cfg.YOLO_DIV, 128//cfg.YOLO_DIV, 256//cfg.YOLO_DIV) # 52x52x256

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256//cfg.YOLO_DIV, 512//cfg.YOLO_DIV), downsample=True) # 26x26x512

    for i in range(1):
        input_data = common.residual_block(input_data, 512//cfg.YOLO_DIV, 256//cfg.YOLO_DIV, 512//cfg.YOLO_DIV) # 26x26x512

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512//cfg.YOLO_DIV, 1024//cfg.YOLO_DIV), downsample=True) # 13x13x1024

    for i in range(1):
        input_data = common.residual_block(input_data, 1024//cfg.YOLO_DIV, 512//cfg.YOLO_DIV, 1024//cfg.YOLO_DIV) # 13x13x1024

    return route_1, route_2, input_data

def YOLOv3(input_layer):
    route_1, route_2, conv = darknet53(input_layer)
    # 52x52x256, 26x26x512, 13x13x1024
    conv = common.convolutional(conv, (1, 1, 1024//cfg.YOLO_DIV,  512//cfg.YOLO_DIV)) # 13x13x512
    conv = common.convolutional(conv, (3, 3,  512//cfg.YOLO_DIV, 1024//cfg.YOLO_DIV)) # 13x13x1024
    # conv = common.convolutional(conv, (1, 1, 1024//cfg.YOLO_DIV,  512//cfg.YOLO_DIV)) # 13x13x512
    # conv = common.convolutional(conv, (3, 3,  512//cfg.YOLO_DIV, 1024//cfg.YOLO_DIV)) # 13x13x1024
    conv = common.convolutional(conv, (1, 1, 1024//cfg.YOLO_DIV,  512//cfg.YOLO_DIV)) # 13x13x512

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512//cfg.YOLO_DIV, 1024//cfg.YOLO_DIV)) # 13x13x1024
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024//cfg.YOLO_DIV, 3*(NUM_CLASS + 5)), activate=False, bn=False) # 13x13x(3*(NUM_CLASS+5)) distcode

    conv = common.convolutional(conv, (1, 1,  512//cfg.YOLO_DIV,  256//cfg.YOLO_DIV)) # 13x13x256
    conv = common.upsample(conv) # 26x26x256

    conv = tf.concat([conv, route_2], axis=-1) # 26x26x768

    conv = common.convolutional(conv, (1, 1, 768//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 26x26x256
    conv = common.convolutional(conv, (3, 3, 256//cfg.YOLO_DIV, 512//cfg.YOLO_DIV)) # 26x26x512
    # conv = common.convolutional(conv, (1, 1, 512//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 26x26x256
    # conv = common.convolutional(conv, (3, 3, 256//cfg.YOLO_DIV, 512//cfg.YOLO_DIV)) # 26x26x512
    conv = common.convolutional(conv, (1, 1, 512//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 26x26x256

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256//cfg.YOLO_DIV, 512//cfg.YOLO_DIV)) # 26x26x512
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512//cfg.YOLO_DIV, 3*(NUM_CLASS + 5)), activate=False, bn=False) # 26x26x(3*(NUM_CLASS+5)) distcode

    conv = common.convolutional(conv, (1, 1, 256//cfg.YOLO_DIV, 128//cfg.YOLO_DIV)) # 26x26x128
    conv = common.upsample(conv) # 52x52x128

    conv = tf.concat([conv, route_1], axis=-1) # 52x52x384

    conv = common.convolutional(conv, (1, 1, 384//cfg.YOLO_DIV, 128//cfg.YOLO_DIV)) # 52x52x128
    conv = common.convolutional(conv, (3, 3, 128//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 52x52x256
    # conv = common.convolutional(conv, (1, 1, 256//cfg.YOLO_DIV, 128//cfg.YOLO_DIV)) # 52x52x128
    # conv = common.convolutional(conv, (3, 3, 128//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 52x52x256
    conv = common.convolutional(conv, (1, 1, 256//cfg.YOLO_DIV, 128//cfg.YOLO_DIV)) # 52x52x128

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128//cfg.YOLO_DIV, 256//cfg.YOLO_DIV)) # 52x52x256
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256//cfg.YOLO_DIV, 3*(NUM_CLASS +5)), activate=False, bn=False) # 52x52x(3*(NUM_CLASS+5)) distcode

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS)) # distcode

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    # conv_raw_prob = conv_output[:, :, :, :, 5:6] # distcode
    # conv_raw_depth= conv_output[:, :, :, :, 6:7]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    # return tf.concat([pred_xywh, pred_conf, pred_prob, conv_raw_depth], axis=-1) # distcode

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / tf.maximum(union_area, 1e-12)

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-12)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / tf.maximum(enclose_area, 1e-12)

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):

    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS)) # distcode

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    # conv_raw_prob = conv[:, :, :, :, 5:6] # distcode

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]
    # pred_raw_depth= pred[:, :, :, :, 6:7] # distcode

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]
    # label_prob    = label[:, :, :, :, 5:6] # distcode
    # label_depth   = label[:, :, :, :, 6:7]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # depth_loss= respond_bbox * tf.abs(pred_raw_depth - label_depth)/100.
    # prob_loss += depth_loss # distcode

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss