from pathlib import Path
from PIL import Image
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from controller import dataline

modelDir = Path('/Data/models/FlowClass2e340/')
# modelDir = Path('/Data/models/UNet3/')

dataDir = Path('/Data/dataset-OrigFiles/')
testFile = Path('/Data/DSyolo/test_RGB_GTs.txt')
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

time()
times = []
for line in lines:
    seq, imageID = extractLineData(line)

    # The next snippet is used to grab the input, run the model, and save the output as a grayscale png
    x = data.singleImage(dataDir/seq/'flow', imageID)
    # x = data.singleImage(dataDir/seq/'frames', imageID)
    x = np.expand_dims(x, 0)
    
    t1 = time()
    network.predict(x)
    t2 = time()
    times.append(t2-t1)
    print(t2-t1)
print(np.average(times))
