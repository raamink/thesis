from pathlib import Path
from PIL import Image
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from controller import dataline

modelDir = Path('/Data/models/FlowClass2e340/')
# modelDir = Path('/Data/models/UNet-RGB-20221108-114619')
inputDir = Path('/Data/models/inputs')

dataDir = Path('/Data/dataset-OrigFiles/')
# testFile = Path('/Data/DSyolo/test_RGB_GTs.txt')
testFile = Path('/Data/dataset-OrigFiles/clip13/corrected_gt_boxes.txt')
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
    
for line in lines:
    seq, imageID = extractLineData(line)
    # if seq in ['clip2', 'clip3', 'clip4vD', 'clip7']:
    #     continue

    """
    # # This snippet turns the test set into 512x512 input samples, as seen by UNet.
    # # It saves the mask, flow, and rgb to /Data/models/inputs

    frame = data.singleImage(dataDir/seq/'frames', imageID)
    out = Image.fromarray((frame*255).astype(np.uint8))
    # out.show()
    out.save(str(inputDir / f'unet-frame-{seq}-{imageID}.png'))

    flow = data.singleImage(dataDir/seq/'flow', imageID)
    flowOut = data.parseFlowEXR_asPolar(flow)

    out = cv2.cvtColor(flowOut, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(inputDir / f'unet-flow-{seq}-{imageID}.png'), out)


    mask = data.singleImage(dataDir/seq/'masks', imageID)
    maskOut = np.squeeze(mask.astype(np.uint8)*255)
    out = Image.fromarray(maskOut)
    # out.show()
    out.save(str(inputDir/f'unet-mask-{seq}-{imageID}.png')) 
    """

    # """
    # # The next snippet is used to grab the input, run the model, and save the output as a grayscale png

    x = data.singleImage(dataDir/seq/'flow', imageID)
    # x = data.singleImage(dataDir/seq/'frames', imageID)
    x = np.expand_dims(x, 0)
    
    pred = network.predict(x)
    
    # predMask = np.round(pred[0,...]).astype(np.uint8)
    # label = data.singleImage(dataDir/seq/'masks', imageID).astype(np.uint8)
    # intersect = np.sum(predMask * label)
    # union = np.sum(predMask) + np.sum(label) - intersect
    # score = (intersect + 1) / (union + 1)
    # print(score)
    # print(f'IoU of {seq}-{imageID}: {score:.4}')

    I8 = (pred[0,...] * 255).astype(np.uint8)

    out = Image.fromarray(I8[...,0],'L')
    out.save(str(outputDir/f'{seq}-{imageID}.png')) 
    # plt.imsave(str(outputDir / f'{seq}-{imageID}.png'), pred[0,...], cmap='gray')

    # """
    # Identify which was just saved.
    print(f'Saving {seq}-{imageID}')