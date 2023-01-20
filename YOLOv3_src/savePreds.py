import cv2
import numpy as np
import utils as utils
import tensorflow as tf
from yolov3 import YOLOv3, decode
from PIL import Image
import OpenEXR
import Imath
import array
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import Dataset


modelDir = Path("/Data/models/22.05.30")
modelWeights = modelDir / 'rgb_yolo'
# modelDir = Path("/Data/models/dec02")
# modelWeights = modelDir / 'yolov3'
dataDir = Path("/Data/dataset-OrigFiles")
# labelsPath = Path(f"/Data/DSyolo/test_RGB_GTs.txt")
labelsPath = Path(f"/Data/dataset-purple/RGB_GTs.txt")
with labelsPath.open() as f:
    lines = f.readlines()
input_size = 416
batch_size = 1

testset = Dataset('test')

outputDir = modelDir / 'predictions'
if not outputDir.exists():
    outputDir.mkdir()

def extractLineData(line):
        fpath, *_ = line.split(' ')
        components = fpath.split('/')
        seq = components[2]
        if seq == 'clip2v2':
            seq = 'clip2'
        imageID = components[-1].split('.')[0]
        return seq,imageID

def returnEXR(exr_path):
    file = OpenEXR.InputFile(str(exr_path))

    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
    img = np.zeros((sz[1],sz[0],3), np.float64)
    img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
    img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)
    img[np.abs(img)>200] = 0
    return img

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights(str(modelWeights))
# model = tf.keras.models.load_model(str(modelDir))

testset.batch_size = 1
testset.num_batchs = 160
for line in lines:
    seq, imageID = extractLineData(line)
    inputdata, _ = testset.__next__()

    # flowPath = dataDir / seq / 'flow' / f'flow{imageID}.exr'
    # exrs = [returnEXR(str(flowPath))]

    pngPath = dataDir / seq / 'frames' / f'{imageID}.png'
    pngs = [cv2.cvtColor(cv2.imread(str(pngPath)), cv2.COLOR_BGR2RGB)]
    
    # inputdata = np.concatenate([utils.image_preporcess(exr, [input_size, input_size])[np.newaxis,...].astype(np.float32) for exr in exrs])
    # inputdata = np.concatenate([utils.image_preporcess(png.copy(), [input_size, input_size])[np.newaxis,...].astype(np.float32) for png in pngs])
    
    predpngs = [cv2.resize(png, (1280, 720)) for png in pngs]
    pred_bbox = model.predict(inputdata) #, training=False)
    bboxes = []
    for i in range(1):
        pbox = [pred_bbox[0][i,:], pred_bbox[1][i,:], pred_bbox[2][i,:]]
        pbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pbox]
        pbox = tf.concat(pbox, axis=0)
        box = utils.postprocess_boxes(pbox, (720, 1280), input_size, 0.3)
        box = utils.nms(box, 0.25, method='nms')
        bboxes.append(box)

    if box != []:
        print(f'ping!\t{imageID} found {len(bboxes[0])}')

    img = utils.draw_bbox(predpngs[0], bboxes[0])
    
    # plt.imshow(img)
    # plt.show()
    
    out = Image.fromarray(img)
    out.save(str(outputDir/f'{seq}-{imageID}.png'))
