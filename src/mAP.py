from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from controller import dataline

modelDir = Path('/Data/models/FlowClass2e340/')
# modelDir = Path('/Data/models/UNet3/')

dataDir = Path('/Data/dataset-farneOF')
testFile = Path('/Data/dataset-farneOF/RGB_GTs.txt')
# dataDir = Path('/Data/dataset-purple')
# testFile = Path('/Data/dataset-purple/RGB_GTs.txt')
# dataDir = Path('/Data/dataset-OrigFiles/')
# testFile = Path('/Data/DSyolo/test_RGB_GTs.txt')
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

outputDir = modelDir / 'predictions'
if not outputDir.exists():
    outputDir.mkdir()

batchSize = 1
data = dataline(dataDir, batchMode='random', batchSize=batchSize)

network = load_model(modelDir)

mAP_scores = []
clip_mAPs = {}
currentSeq = ''
for line in lines:
    seq, imageID = extractLineData(line)

    if seq == currentSeq:
        pass
    elif currentSeq != '':
        clip_mAPs[currentSeq] = np.average(clip_mAPs[currentSeq])
        print(f'Clip {currentSeq} average IoU: {clip_mAPs[currentSeq]:.4}')
        clip_mAPs[seq] = []
        currentSeq = seq
    else:
        clip_mAPs[seq] = []
        currentSeq = seq


    # The next snippet is used to grab the input, run the model, and save the output as a grayscale png
    x = data.singleImage(dataDir/seq/'flow', imageID)
    # x = data.singleImage(dataDir/seq/'frames', imageID)
    x = np.expand_dims(x, 0)
    
    
    pred = network.predict(x)
    label = data.singleImage(dataDir/seq/'masks', imageID).astype(np.bool)
    
    thresholds = np.arange(0,1.01,0.01)
    ps = []
    rs = []
    for thresh in thresholds:
        predMask = pred[0,...] >= thresh
        invPred = np.invert(predMask)
        invLabel = np.invert(label)

        truePositiveMask = np.bitwise_and(predMask, label)
        falsePositiveMask = np.bitwise_xor(predMask, truePositiveMask)
        trueNegativeMask = np.bitwise_and(invPred, invLabel)
        falseNegativeMask = np.bitwise_xor(invPred, trueNegativeMask)

        tp = np.sum(truePositiveMask)
        fp = np.sum(falsePositiveMask)
        # tn = np.sum(trueNegativeMask)
        fn = np.sum(falseNegativeMask)

        precision = tp / (tp + fp + 1)
        recall = tp / (tp + fn + 1)
        
        ps.append(precision)
        rs.append(recall)

    rs.append(0)
    ps.append(1)

    rs = np.array(rs)
    ps = np.array(ps)
    
    averagePrecision =  np.sum((rs[:-1] - rs[1:]) * ps[:-1])
    score = averagePrecision
    
    # print(score)
    print(f'AP of {seq}-{imageID}: {score:.4}')
    mAP_scores.append(score)
    clip_mAPs[seq].append(score)

    # I8 = (pred[0,...] * 255).astype(np.uint8)

    # out = Image.fromarray(I8[...,0],'L')
    # out.show()
clip_mAPs[seq] = np.average(clip_mAPs[seq])
print(f'Average score over dataset: {np.average(mAP_scores):.4}')
for k, v in clip_mAPs.items():
    print(f'{k}:\t{v:.4}')

    