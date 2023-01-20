YOLO_CLASSES     = {0: "crazyflie"}
YOLO_ANCHORS     = [[[1.25,1.625], [2.0,3.75], [4.125,2.875]], [[1.875,3.8125], [3.875,2.8125], [3.6875,7.4375]], [[3.625,2.8125], [4.875,6.1875], [11.65625,10.1875]]]
YOLO_STRIDES     = [8, 16, 32]
YOLO_ANCHOR_PER_SCALE    = 3
YOLO_IOU_LOSS_THRESH     = 0.5
YOLO_DIV = 4

# TRAIN_ANNOT_PATH          = "./dataset/box_train.txt"
TRAIN_ANNOT_PATH          = "./dataset/gt_boxes.txt"
TRAIN_BATCH_SIZE          = 10
TRAIN_INPUT_SIZE          = [416]
TRAIN_LR_INIT             = 1e-3
TRAIN_LR_END              = 1e-6
TRAIN_WARMUP_EPOCHS       = 2
TRAIN_EPOCHS              = 10

TEST_ANNOT_PATH           = "./dataset/box_test.txt"
TEST_BATCH_SIZE           = 10
TEST_INPUT_SIZE           = [416]
TEST_DECTECTED_IMAGE_PATH = "./test"
TEST_SCORE_THRESHOLD      = 0.3
TEST_IOU_THRESHOLD        = 0.45
