from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from controller import dataline

modelDir = Path('/Data/models/FlowClass2e340/')
# modelDir = Path('/Data/models/UNet-RGB-20221107-161453')

# dataDir = Path('/Data/dataset-OrigFiles/')
dataDir = Path('/Data/dataset-purple')
testFile = Path('/Data/dataset-purple/RGB_GTs.txt')
with testFile.open() as f:
    lines = f.readlines()

def extractLineData(line):
    fpath, *_ = line.split(' ')
    components = fpath.split('/')
    seq = components[2]
    if seq == 'clip2v2':
        seq = 'clip2'
    imageID = int(components[-1].split('.')[0])
    return seq,imageID

batchSize = 1
data = dataline(dataDir, batchMode='random', batchSize=batchSize)

network = load_model(modelDir)

iou_scores = []
clip_ious = {}
currentSeq = ''
for line in lines:
    seq, imageID = extractLineData(line)

    if seq == currentSeq:
        pass
    elif currentSeq != '':
        clip_ious[currentSeq] = np.average(clip_ious[currentSeq])
        print(f'Clip {currentSeq} average IoU: {clip_ious[currentSeq]:.4}')
        clip_ious[seq] = []
        currentSeq = seq
    else:
        clip_ious[seq] = []
        currentSeq = seq


    # The next snippet is used to grab the input, run the model, and save the output as a grayscale png
    x = data.singleImage(dataDir/seq/'flow', imageID)
    # x = data.singleImage(dataDir/seq/'frames', imageID)
    x = np.expand_dims(x, 0)
    
    
    pred = network.predict(x)
    
    predMask = np.round(pred[0,...]).astype(np.uint8)
    label = data.singleImage(dataDir/seq/'masks', imageID).astype(np.uint8)
    intersect = np.sum(predMask * label)
    union = np.sum(predMask) + np.sum(label) - intersect
    score = (intersect + 1) / (union + 1)
    # print(score)
    print(f'IoU of {seq}-{imageID}: {score:.4}')
    iou_scores.append(score)
    clip_ious[seq].append(score)

    # I8 = (pred[0,...] * 255).astype(np.uint8)

    # out = Image.fromarray(I8[...,0],'L')
    # out.show()
clip_ious[seq] = np.average(clip_ious[seq])
print(f'Average score over dataset: {np.average(iou_scores):.4}')
for k, v in clip_ious.items():
    print(f'{k}:\t{v:.4}')

    